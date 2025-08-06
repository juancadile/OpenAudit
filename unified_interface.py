"""
OpenAudit Unified Interface
Research-grade bias testing platform combining historical analysis and live experiments
"""

import glob
import json
import os
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit

from core.bias_testing_framework import BiasDatasets
from core.decorators import api_error_handler, log_performance, validate_json_request
from core.logging_config import (
    configure_external_loggers,
    get_logger,
    log_experiment_event,
    setup_logging,
)
from core.model_manager import ModelManager

# Import existing modules with updated paths
from core.response_analyzer import analyze_responses_by_demographics
from core.template_manager import TemplateManager

# Initialize logging
setup_logging()
configure_external_loggers()
logger = get_logger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("OPENAUDIT_SECRET_KEY", "openaudit-unified-secret")

# Configure SocketIO with more robust settings
try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
        async_mode="threading",
    )
    logger.info("Socket.IO initialized successfully")
except Exception as e:
    logger.error(f"Socket.IO initialization failed: {e}")
    # Fallback to regular Flask without Socket.IO
    socketio = None

# Initialize Google-level management systems
logger.info("Initializing OpenAudit management systems...")
template_manager = TemplateManager()
model_manager = ModelManager()
logger.info("Template and Model managers initialized")

# Global state for live experiments
live_experiments = {}
current_experiment = None

# ============================================================================
# SHARED DATA MANAGEMENT
# ============================================================================


def get_available_runs():
    """Get all available experiment runs (shared by both interfaces)"""
    runs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")

    if os.path.exists(runs_dir):
        pattern1 = os.path.join(runs_dir, "hiring_bias_experiment_*.json")
        pattern2 = os.path.join(runs_dir, "live_experiment_*.json")
        files = glob.glob(pattern1) + glob.glob(pattern2)
    else:
        files = []

    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".json", "").split("_")

        if filename.startswith("hiring_bias_experiment_") and len(parts) >= 4:
            date_part = parts[-2]
            time_part = parts[-1]
            timestamp_str = f"{date_part}_{time_part}"
            experiment_type = "Hiring Bias"
        elif filename.startswith("live_experiment_") and len(parts) >= 3:
            if len(parts[-1]) == 6 and parts[-1].isdigit():
                date_part = parts[-2]
                time_part = parts[-1]
                timestamp_str = f"{date_part}_{time_part}"
                experiment_type = "Live Experiment"
            else:
                continue
        else:
            continue

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            runs.append(
                {
                    "filepath": filepath,
                    "filename": filename,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_raw": timestamp_str,
                    "experiment_type": experiment_type,
                }
            )
        except ValueError:
            continue

    return sorted(runs, key=lambda x: x["timestamp_raw"], reverse=True)


# ============================================================================
# ROUTES - MAIN INTERFACE
# ============================================================================


@app.route("/")
def index():
    """Main unified dashboard"""
    runs = get_available_runs()
    return render_template("unified_dashboard.html", runs=runs)


# ============================================================================
# ROUTES - HISTORICAL ANALYSIS API
# ============================================================================


@app.route("/api/runs")
def api_runs():
    """API endpoint to get available runs"""
    logger.debug("Fetching available runs")
    runs = get_available_runs()
    logger.info(f"Found {len(runs)} available runs")
    return jsonify(runs)


@app.route("/api/analyze/<run_id>")
def api_analyze(run_id):
    """API endpoint to analyze a specific run"""
    logger.info(f"Analyzing run: {run_id}")
    runs = get_available_runs()
    run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

    if not run:
        logger.warning(f"Run not found: {run_id}")
        return jsonify({"error": "Run not found"}), 404

    try:
        # Load and basic analysis
        with open(run["filepath"], "r") as f:
            raw_data = json.load(f)

        # Handle both data formats
        if isinstance(raw_data, dict) and "responses" in raw_data:
            data = raw_data["responses"]
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            return jsonify({"error": "Unknown data format"}), 500

        # Analyze by demographics
        demographic_results = analyze_responses_by_demographics(data)

        # Format for web display
        analysis = {
            "total_responses": len(data),
            "models": list(set(item["model_name"] for item in data)),
            "demographics": {},
            "models_analysis": {},
            "bias_summary": {},
        }

        # Demographics analysis
        for demo, model_data in demographic_results.items():
            all_decisions = []
            for model, responses in model_data.items():
                decisions = [r["decision"] for r in responses]
                all_decisions.extend(decisions)

            yes_count = all_decisions.count("YES")
            total_count = len(all_decisions)
            hire_rate = yes_count / total_count if total_count > 0 else 0

            analysis["demographics"][demo] = {
                "hire_rate": hire_rate,
                "yes_count": yes_count,
                "total_count": total_count,
                "hire_rate_percent": f"{hire_rate*100:.1f}%",
            }

        # Model analysis
        for model in analysis["models"]:
            model_decisions = []
            model_demographics = {}

            for demo, model_data in demographic_results.items():
                if model in model_data:
                    decisions = [r["decision"] for r in model_data[model]]
                    yes_count = decisions.count("YES")
                    total_count = len(decisions)
                    hire_rate = yes_count / total_count if total_count > 0 else 0

                    model_demographics[demo] = {
                        "hire_rate": hire_rate,
                        "yes_count": yes_count,
                        "total_count": total_count,
                    }
                    model_decisions.extend(decisions)

            overall_yes = model_decisions.count("YES")
            overall_total = len(model_decisions)
            overall_rate = overall_yes / overall_total if overall_total > 0 else 0

            analysis["models_analysis"][model] = {
                "overall_hire_rate": overall_rate,
                "overall_hire_rate_percent": f"{overall_rate*100:.1f}%",
                "demographics": model_demographics,
            }

        # Bias summary
        if analysis["demographics"]:
            hire_rates = [
                info["hire_rate"] for info in analysis["demographics"].values()
            ]
            max_rate = max(hire_rates)
            min_rate = min(hire_rates)
            bias_gap = max_rate - min_rate

            max_demo = max(
                analysis["demographics"].items(), key=lambda x: x[1]["hire_rate"]
            )
            min_demo = min(
                analysis["demographics"].items(), key=lambda x: x[1]["hire_rate"]
            )

            analysis["bias_summary"] = {
                "bias_gap": bias_gap,
                "bias_gap_percent": f"{bias_gap*100:.1f}%",
                "max_demo": max_demo[0],
                "max_rate": f"{max_demo[1]['hire_rate']*100:.1f}%",
                "min_demo": min_demo[0],
                "min_rate": f"{min_demo[1]['hire_rate']*100:.1f}%",
                "significant_bias": bias_gap > 0.1,
            }

        analysis["run_info"] = run
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/aggregate")
def api_aggregate():
    """API endpoint to aggregate multiple runs"""
    run_ids = request.args.get("runs", "").split(",")
    runs = get_available_runs()

    selected_runs = [r for r in runs if r["timestamp_raw"] in run_ids]

    if not selected_runs:
        return jsonify({"error": "No valid runs selected"}), 400

    # Aggregate data from multiple runs
    aggregated_data = []
    for run in selected_runs:
        try:
            with open(run["filepath"], "r") as f:
                raw_data = json.load(f)

            # Handle both data formats
            if isinstance(raw_data, dict) and "responses" in raw_data:
                # Live experiment format: extract responses array
                data = raw_data["responses"]
            elif isinstance(raw_data, list):
                # Original CLI format: data is already the responses array
                data = raw_data
            else:
                logger.warning(
                    f"Unknown data format in file {run['filepath']}, skipping"
                )
                continue

            aggregated_data.extend(data)
        except Exception as e:
            logger.error(f"Error loading {run['filepath']}: {e}")
            continue

    if not aggregated_data:
        return jsonify({"error": "No data to aggregate"}), 500

    # Analyze aggregated data
    demographic_results = analyze_responses_by_demographics(aggregated_data)

    # Format similar to single run analysis
    analysis = {
        "total_responses": len(aggregated_data),
        "models": list(set(item["model_name"] for item in aggregated_data)),
        "runs_aggregated": len(selected_runs),
        "demographics": {},
        "models_analysis": {},
        "bias_summary": {},
    }

    # Same analysis logic as single run
    for demo, model_data in demographic_results.items():
        all_decisions = []
        for model, responses in model_data.items():
            decisions = [r["decision"] for r in responses]
            all_decisions.extend(decisions)

        yes_count = all_decisions.count("YES")
        total_count = len(all_decisions)
        hire_rate = yes_count / total_count if total_count > 0 else 0

        analysis["demographics"][demo] = {
            "hire_rate": hire_rate,
            "yes_count": yes_count,
            "total_count": total_count,
            "hire_rate_percent": f"{hire_rate*100:.1f}%",
        }

    # Model analysis
    for model in analysis["models"]:
        model_decisions = []
        model_demographics = {}

        for demo, model_data in demographic_results.items():
            if model in model_data:
                decisions = [r["decision"] for r in model_data[model]]
                yes_count = decisions.count("YES")
                total_count = len(decisions)
                hire_rate = yes_count / total_count if total_count > 0 else 0

                model_demographics[demo] = {
                    "hire_rate": hire_rate,
                    "yes_count": yes_count,
                    "total_count": total_count,
                }
                model_decisions.extend(decisions)

        overall_yes = model_decisions.count("YES")
        overall_total = len(model_decisions)
        overall_rate = overall_yes / overall_total if overall_total > 0 else 0

        analysis["models_analysis"][model] = {
            "overall_hire_rate": overall_rate,
            "overall_hire_rate_percent": f"{overall_rate*100:.1f}%",
            "demographics": model_demographics,
        }

    # Bias summary
    if analysis["demographics"]:
        hire_rates = [info["hire_rate"] for info in analysis["demographics"].values()]
        max_rate = max(hire_rates)
        min_rate = min(hire_rates)
        bias_gap = max_rate - min_rate

        max_demo = max(
            analysis["demographics"].items(), key=lambda x: x[1]["hire_rate"]
        )
        min_demo = min(
            analysis["demographics"].items(), key=lambda x: x[1]["hire_rate"]
        )

        analysis["bias_summary"] = {
            "bias_gap": bias_gap,
            "bias_gap_percent": f"{bias_gap*100:.1f}%",
            "max_demo": max_demo[0],
            "max_rate": f"{max_demo[1]['hire_rate']*100:.1f}%",
            "min_demo": min_demo[0],
            "min_rate": f"{min_demo[1]['hire_rate']*100:.1f}%",
            "significant_bias": bias_gap > 0.1,
        }

    return jsonify(analysis)


@app.route("/api/comprehensive_analysis/<run_id>")
def api_comprehensive_analysis(run_id):
    """API endpoint for comprehensive statistical bias analysis"""
    runs = get_available_runs()
    run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

    if not run:
        return jsonify({"error": "Run not found"}), 404

    try:
        # Load and prepare data
        with open(run["filepath"], "r") as f:
            raw_data = json.load(f)

        # Handle both data formats
        if isinstance(raw_data, dict) and "responses" in raw_data:
            data = raw_data["responses"]
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            return jsonify({"error": "Unknown data format"}), 500

        # Convert to analyzer format
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse

        llm_responses = []
        for item in data:
            response = LLMResponse(
                model_name=item.get("model_name", "unknown"),
                provider=item.get("provider", "unknown"),
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                timestamp=item.get("timestamp", ""),
                metadata=item.get("metadata", {}),
            )
            llm_responses.append(response)

        # Perform comprehensive analysis
        analyzer = BiasAnalyzer(llm_responses)
        comprehensive_analysis = analyzer.comprehensive_bias_analysis()

        # Add run info
        comprehensive_analysis["run_info"] = run

        return jsonify(comprehensive_analysis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTES - MODULAR ANALYSIS API
# ============================================================================


@app.route("/api/modules")
@api_error_handler
@log_performance()
def api_get_available_modules():
    """Get all available analysis modules"""
    try:
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse
        from core.module_registry import get_global_registry

        # Create a temporary analyzer to access module registry
        temp_responses = [LLMResponse("temp", "temp", "temp", "temp", "", {})]
        analyzer = BiasAnalyzer(temp_responses)

        available_modules = analyzer.get_available_modules()

        # Get detailed module information
        module_details = {}
        for module_name in available_modules:
            registry = get_global_registry()
            module_info = registry.get_module_info(module_name)
            if module_info:
                module_details[module_name] = module_info
            else:
                module_details[module_name] = {
                    "name": module_name,
                    "description": f"Analysis module: {module_name}",
                    "version": "1.0.0",
                    "dependencies": [],
                    "capabilities": ["bias_analysis"],
                }

        return jsonify(
            {
                "available_modules": available_modules,
                "module_details": module_details,
                "total_modules": len(available_modules),
            }
        )

    except Exception as e:
        logger.error(f"Error getting available modules: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles")
@api_error_handler
@log_performance()
def api_get_available_profiles():
    """Get all available analysis profiles"""
    try:
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse

        # Create a temporary analyzer to access profiles
        temp_responses = [LLMResponse("temp", "temp", "temp", "temp", "", {})]
        analyzer = BiasAnalyzer(temp_responses)

        available_profiles = analyzer.get_available_profiles()

        return jsonify(
            {
                "available_profiles": available_profiles,
                "total_profiles": len(available_profiles),
            }
        )

    except Exception as e:
        logger.error(f"Error getting available profiles: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modular_analysis/<run_id>")
@api_error_handler
@log_performance()
def api_modular_analysis(run_id):
    """Run modular bias analysis on a specific run"""
    try:
        # Get request parameters
        selected_modules = request.args.getlist("modules")
        profile = request.args.get("profile")
        alpha = float(request.args.get("alpha", 0.05))
        effect_size_threshold = float(request.args.get("effect_size_threshold", 0.1))
        control_group = request.args.get("control_group")

        # Find the run
        runs = get_available_runs()
        run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

        if not run:
            return jsonify({"error": "Run not found"}), 404

        # Load and prepare data
        with open(run["filepath"], "r") as f:
            raw_data = json.load(f)

        # Handle both data formats
        if isinstance(raw_data, dict) and "responses" in raw_data:
            data = raw_data["responses"]
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            return jsonify({"error": "Unknown data format"}), 500

        # Convert to analyzer format
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse

        llm_responses = []
        for item in data:
            response = LLMResponse(
                model_name=item.get("model_name", "unknown"),
                provider=item.get("provider", "unknown"),
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                timestamp=item.get("timestamp", ""),
                metadata=item.get("metadata", {}),
            )
            llm_responses.append(response)

        # Perform modular analysis
        analyzer = BiasAnalyzer(llm_responses)

        # Prepare analysis parameters
        analysis_kwargs = {
            "alpha": alpha,
            "effect_size_threshold": effect_size_threshold,
        }

        if control_group:
            analysis_kwargs["control_group"] = control_group

        # Run modular analysis
        if profile:
            # Use predefined profile
            modular_analysis = analyzer.run_profile_analysis(profile, **analysis_kwargs)
        elif selected_modules:
            # Use custom module selection
            modular_analysis = analyzer.run_modular_analysis(
                selected_modules=selected_modules, **analysis_kwargs
            )
        else:
            # Use default comprehensive profile
            modular_analysis = analyzer.run_modular_analysis(
                profile="comprehensive", **analysis_kwargs
            )

        # Add run info and analysis metadata
        modular_analysis["run_info"] = run
        modular_analysis["analysis_parameters"] = {
            "selected_modules": selected_modules or "profile_based",
            "profile": profile,
            "alpha": alpha,
            "effect_size_threshold": effect_size_threshold,
            "control_group": control_group,
        }

        return jsonify(modular_analysis)

    except Exception as e:
        logger.error(f"Error in modular analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/module/<module_name>")
@api_error_handler
@log_performance()
def api_get_module_info(module_name):
    """Get detailed information about a specific module"""
    try:
        from core.module_registry import get_global_registry

        registry = get_global_registry()

        if not registry.has_module(module_name):
            return jsonify({"error": "Module not found"}), 404

        module_info = registry.get_module_info(module_name)
        compatibility = registry.check_module_compatibility(module_name)

        return jsonify(
            {
                "module_info": module_info,
                "compatibility": compatibility,
                "is_available": registry.has_module(module_name),
            }
        )

    except Exception as e:
        logger.error(f"Error getting module info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_custom_analysis", methods=["POST"])
@api_error_handler
@validate_json_request
@log_performance()
def api_run_custom_analysis():
    """Run custom modular analysis with uploaded data or configuration"""
    try:
        data = request.get_json()

        # Validate required fields
        if "run_id" not in data:
            return jsonify({"error": "run_id is required"}), 400

        run_id = data["run_id"]
        selected_modules = data.get("modules", [])
        profile = data.get("profile")
        analysis_config = data.get("config", {})

        # Load run data
        runs = get_available_runs()
        run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

        if not run:
            return jsonify({"error": "Run not found"}), 404

        with open(run["filepath"], "r") as f:
            raw_data = json.load(f)

        # Handle both data formats
        if isinstance(raw_data, dict) and "responses" in raw_data:
            responses_data = raw_data["responses"]
        elif isinstance(raw_data, list):
            responses_data = raw_data
        else:
            return jsonify({"error": "Unknown data format"}), 500

        # Convert to analyzer format
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse

        llm_responses = []
        for item in responses_data:
            response = LLMResponse(
                model_name=item.get("model_name", "unknown"),
                provider=item.get("provider", "unknown"),
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                timestamp=item.get("timestamp", ""),
                metadata=item.get("metadata", {}),
            )
            llm_responses.append(response)

        # Perform custom modular analysis
        analyzer = BiasAnalyzer(llm_responses)

        # Merge default config with provided config
        default_config = {
            "alpha": 0.05,
            "effect_size_threshold": 0.1,
            "confidence_level": 0.95,
        }
        merged_config = {**default_config, **analysis_config}

        # Run analysis
        if profile:
            analysis_result = analyzer.run_profile_analysis(profile, **merged_config)
        elif selected_modules:
            analysis_result = analyzer.run_modular_analysis(
                selected_modules=selected_modules, **merged_config
            )
        else:
            # Default to standard profile
            analysis_result = analyzer.run_profile_analysis("standard", **merged_config)

        # Add metadata
        analysis_result["custom_analysis_info"] = {
            "run_id": run_id,
            "modules_used": selected_modules or "profile_based",
            "profile_used": profile,
            "custom_config": analysis_config,
            "merged_config": merged_config,
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error in custom analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/module_compatibility")
@api_error_handler
@log_performance()
def api_check_module_compatibility():
    """Check compatibility between selected modules"""
    try:
        selected_modules = request.args.getlist("modules")

        if not selected_modules:
            return jsonify({"error": "No modules specified"}), 400

        from core.module_registry import get_global_registry

        registry = get_global_registry()
        compatibility = registry.validate_module_compatibility(selected_modules)

        # Add individual module info
        module_details = {}
        for module_name in selected_modules:
            if registry.has_module(module_name):
                module_details[module_name] = registry.get_module_info(module_name)
            else:
                module_details[module_name] = {"error": "Module not found"}

        return jsonify(
            {
                "compatibility": compatibility,
                "selected_modules": selected_modules,
                "module_details": module_details,
            }
        )

    except Exception as e:
        logger.error(f"Error checking module compatibility: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/enhanced_analysis/<run_id>")
@api_error_handler
@log_performance()
def api_enhanced_comprehensive_analysis(run_id):
    """Run enhanced comprehensive analysis using all available modules"""
    try:
        # Find the run
        runs = get_available_runs()
        run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

        if not run:
            return jsonify({"error": "Run not found"}), 404

        # Load and prepare data
        with open(run["filepath"], "r") as f:
            raw_data = json.load(f)

        # Handle both data formats
        if isinstance(raw_data, dict) and "responses" in raw_data:
            data = raw_data["responses"]
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            return jsonify({"error": "Unknown data format"}), 500

        # Convert to analyzer format
        from core.bias_testing_framework import BiasAnalyzer, LLMResponse

        llm_responses = []
        for item in data:
            response = LLMResponse(
                model_name=item.get("model_name", "unknown"),
                provider=item.get("provider", "unknown"),
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                timestamp=item.get("timestamp", ""),
                metadata=item.get("metadata", {}),
            )
            llm_responses.append(response)

        # Perform enhanced comprehensive analysis
        analyzer = BiasAnalyzer(llm_responses)

        # Use async enhanced analysis
        import asyncio

        enhanced_analysis = asyncio.run(analyzer.enhanced_comprehensive_analysis())

        # Add run info
        enhanced_analysis["run_info"] = run

        return jsonify(enhanced_analysis)

    except Exception as e:
        logger.error(f"Error in enhanced comprehensive analysis: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTES - LIVE EXPERIMENT API
# ============================================================================


@app.route("/api/datasets")
def api_datasets():
    """Get available bias datasets"""
    bias_datasets = BiasDatasets()
    return jsonify(
        {
            "demographics": bias_datasets.get_demographics(),
            "cv_levels": ["weak", "borderline", "strong"],
        }
    )


# ============================================================================
# TEMPLATE & MODEL MANAGEMENT APIs
# ============================================================================


@app.route("/api/cv_templates")
@api_error_handler
@log_performance()
def get_cv_templates():
    """Get all available CV templates"""
    templates = template_manager.get_cv_templates()

    result = [
        {
            "name": t.name,
            "role": t.role,
            "level": t.level,
            "description": t.description,
            "template": t.template,
            "variables": t.variables,
            "author": t.author,
            "updated_at": t.updated_at,
        }
        for t in templates
    ]

    return jsonify(result)


@app.route("/api/prompt_templates")
def get_prompt_templates():
    """Get all available prompt templates"""
    try:
        templates = template_manager.get_prompt_templates()
        return jsonify(
            [
                {
                    "name": t.name,
                    "category": t.category,
                    "description": t.description,
                    "template": t.template,
                    "variables": t.variables,
                    "bias_focus": t.bias_focus,
                    "author": t.author,
                    "updated_at": t.updated_at,
                }
                for t in templates
            ]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview_cv")
def preview_cv():
    """Preview a CV with specific variables"""
    try:
        # Get parameters
        template_name = request.args.get("template", "Software Engineer - Strong")
        name = request.args.get("name", "Sample Candidate")
        university = request.args.get("university", "Stanford University")
        experience = request.args.get("experience", "5")
        address = request.args.get("address", "123 Main St, San Francisco, CA")

        # Generate CV content
        variables = {
            "name": name,
            "university": university,
            "experience": experience,
            "address": address,
        }

        cv_content = template_manager.generate_cv_content(template_name, variables)

        return jsonify(
            {
                "template_name": template_name,
                "variables": variables,
                "cv_content": cv_content,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview_prompt")
def preview_prompt():
    """Preview a prompt with specific variables"""
    try:
        template_name = request.args.get("template", "Standard Hiring Assessment")
        name = request.args.get("name", "Sample Candidate")
        cv_content = request.args.get("cv_content", "Sample CV content...")

        variables = {"name": name, "cv_content": cv_content}

        prompt_content = template_manager.generate_prompt_content(
            template_name, variables
        )

        return jsonify(
            {
                "template_name": template_name,
                "variables": variables,
                "prompt_content": prompt_content,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def get_available_models():
    """Get all available AI models organized by provider"""
    try:
        models_by_category = model_manager.get_models_by_category()
        all_models = model_manager.get_available_models()

        return jsonify(
            {
                "models_by_category": models_by_category,
                "all_models": all_models,
                "total_count": len(all_models),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/templates/export")
def export_templates():
    """Export all templates to a downloadable file"""
    try:
        export_path = (
            f"templates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        template_manager.export_templates(export_path)

        # Return file for download
        return send_file(export_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/templates/import", methods=["POST"])
def import_templates():
    """Import templates from uploaded file"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file temporarily
        temp_path = f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file.save(temp_path)

        # Import templates
        imported = template_manager.import_templates(
            temp_path, overwrite=request.form.get("overwrite", "false") == "true"
        )

        # Clean up temp file
        os.remove(temp_path)

        return jsonify(
            {
                "imported": imported,
                "message": f"Imported {imported['cv_templates']} CV templates and {imported['prompt_templates']} prompt templates",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TEMPLATE EDITING APIs
# ============================================================================


@app.route("/api/cv_templates/<template_name>")
def get_cv_template(template_name):
    """Get a specific CV template for editing"""
    try:
        template = template_manager.get_cv_template(template_name)
        if not template:
            return jsonify({"error": "Template not found"}), 404

        return jsonify(
            {
                "name": template.name,
                "role": template.role,
                "level": template.level,
                "description": template.description,
                "template": template.template,
                "variables": template.variables,
                "author": template.author,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cv_templates", methods=["POST"])
def create_cv_template():
    """Create a new CV template"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "role", "level", "description", "template"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create template
        template = template_manager.create_cv_template(
            name=data["name"],
            role=data["role"],
            level=data["level"],
            description=data["description"],
            template=data["template"],
            variables=data.get("variables", []),
        )

        return jsonify(
            {
                "message": "CV template created successfully",
                "template": {
                    "name": template.name,
                    "role": template.role,
                    "level": template.level,
                    "description": template.description,
                    "variables": template.variables,
                    "author": template.author,
                    "updated_at": template.updated_at,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cv_templates/<template_name>", methods=["PUT"])
def update_cv_template(template_name):
    """Update an existing CV template"""
    try:
        data = request.get_json()

        # Update template
        template = template_manager.update_cv_template(
            template_name,
            name=data.get("name"),
            role=data.get("role"),
            level=data.get("level"),
            description=data.get("description"),
            template=data.get("template"),
            variables=data.get("variables"),
        )

        if not template:
            return jsonify({"error": "Template not found"}), 404

        return jsonify(
            {
                "message": "CV template updated successfully",
                "template": {
                    "name": template.name,
                    "role": template.role,
                    "level": template.level,
                    "description": template.description,
                    "variables": template.variables,
                    "author": template.author,
                    "updated_at": template.updated_at,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cv_templates/<template_name>", methods=["DELETE"])
def delete_cv_template(template_name):
    """Delete a CV template"""
    try:
        success = template_manager.delete_cv_template(template_name)

        if not success:
            return jsonify({"error": "Template not found"}), 404

        return jsonify({"message": "CV template deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_templates/<template_name>")
def get_prompt_template(template_name):
    """Get a specific prompt template for editing"""
    try:
        template = template_manager.get_prompt_template(template_name)
        if not template:
            return jsonify({"error": "Template not found"}), 404

        return jsonify(
            {
                "name": template.name,
                "category": template.category,
                "description": template.description,
                "template": template.template,
                "variables": template.variables,
                "bias_focus": template.bias_focus,
                "author": template.author,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_templates", methods=["POST"])
def create_prompt_template():
    """Create a new prompt template"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "category", "description", "template"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create template
        template = template_manager.create_prompt_template(
            name=data["name"],
            category=data["category"],
            description=data["description"],
            template=data["template"],
            variables=data.get("variables", []),
            bias_focus=data.get("bias_focus", []),
        )

        return jsonify(
            {
                "message": "Prompt template created successfully",
                "template": {
                    "name": template.name,
                    "category": template.category,
                    "description": template.description,
                    "variables": template.variables,
                    "bias_focus": template.bias_focus,
                    "author": template.author,
                    "updated_at": template.updated_at,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_templates/<template_name>", methods=["PUT"])
def update_prompt_template(template_name):
    """Update an existing prompt template"""
    try:
        data = request.get_json()

        # Update template
        template = template_manager.update_prompt_template(
            template_name,
            name=data.get("name"),
            category=data.get("category"),
            description=data.get("description"),
            template=data.get("template"),
            variables=data.get("variables"),
            bias_focus=data.get("bias_focus"),
        )

        if not template:
            return jsonify({"error": "Template not found"}), 404

        return jsonify(
            {
                "message": "Prompt template updated successfully",
                "template": {
                    "name": template.name,
                    "category": template.category,
                    "description": template.description,
                    "variables": template.variables,
                    "bias_focus": template.bias_focus,
                    "author": template.author,
                    "updated_at": template.updated_at,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_templates/<template_name>", methods=["DELETE"])
def delete_prompt_template(template_name):
    """Delete a prompt template"""
    try:
        success = template_manager.delete_prompt_template(template_name)

        if not success:
            return jsonify({"error": "Template not found"}), 404

        return jsonify({"message": "Prompt template deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SOCKETIO EVENTS - LIVE EXPERIMENTS
# ============================================================================

if socketio:

    @socketio.on("start_experiment")
    def handle_start_experiment(data):
        """Start a new live experiment"""
        if current_experiment:
            emit("experiment_error", {"error": "An experiment is already running"})
            return

        experiment_id = str(uuid.uuid4())
        live_experiments[experiment_id] = {
            "id": experiment_id,
            "config": data,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        emit(
            "experiment_started",
            {"experiment_id": experiment_id, "total_iterations": 10},  # placeholder
        )

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection"""
        emit("connected", {"status": "Connected to OpenAudit Live Experiments"})


# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting OpenAudit Complete Unified Interface")
    logger.info("==================================================")
    logger.info("📊 Historical Analysis: Available")
    if socketio:
        logger.info("🔬 Live Experiments: Available")
        logger.info("🌐 WebSocket Support: Active")
    else:
        logger.warning("🔬 Live Experiments: Limited (No Socket.IO)")
    logger.info("🌐 Dashboard: http://localhost:5100")
    logger.info("==================================================")

    # Log startup event
    log_experiment_event(
        event_type="system_start", socketio_enabled=socketio is not None, port=5100
    )

    debug_mode = os.getenv("OPENAUDIT_DEBUG", "false").lower() == "true"
    if socketio:
        socketio.run(app, debug=debug_mode, port=5100, allow_unsafe_werkzeug=debug_mode)
    else:
        app.run(debug=debug_mode, port=5100)
