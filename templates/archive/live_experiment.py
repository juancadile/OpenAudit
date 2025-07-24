"""
OpenAudit Live Experiment Interface
Real-time prompt visualization and response monitoring
"""

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from core.bias_testing_framework import BiasDatasets, HiringBiasTest
from core.cv_templates import CVTemplates
from core.multi_llm_dispatcher import MultiLLMDispatcher
from core.response_analyzer import analyze_responses_by_demographics

app = Flask(__name__)
app.config["SECRET_KEY"] = "openaudit-live-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for live experiments
live_experiments = {}
current_experiment = None


def create_custom_test_cases(config):
    """Create test cases based on selected demographics and custom prompt"""
    from bias_testing_framework import BiasTestCase

    bias_test = HiringBiasTest()
    role = config.get("role", "software_engineer")
    selected_demographics = config.get("demographics", {})
    custom_prompt = config.get("prompt_template", "")
    custom_cv_template = config.get("custom_cv_template", None)
    cv_level = config.get("cv_level", "borderline")

    # Set custom CV template if provided
    if custom_cv_template and custom_cv_template.get("template"):
        CVTemplates.set_custom_template(
            custom_cv_template["role"],
            custom_cv_template["template"],
            custom_cv_template["cv_level"],
        )

    test_cases = []

    # Use custom prompt if provided, otherwise use default
    if custom_prompt.strip():
        prompt_template = custom_prompt
    else:
        prompt_template = bias_test.role_templates.get(
            role, bias_test.role_templates["software_engineer"]
        )

    if not selected_demographics:
        # Fall back to default test cases with custom prompt
        test_cases = bias_test.create_test_cases(
            role, config.get("cv_level", "borderline")
        )
        # Update all test cases with custom prompt
        for test_case in test_cases:
            test_case.template = prompt_template
        return test_cases

    for demo_key, demo_config in selected_demographics.items():
        if demo_config.get("selected", False):
            custom_names = demo_config.get("names", [])

            if custom_names:
                # Create individual test cases for each name with consistent qualifications
                for name in custom_names:
                    variables = {
                        "name": [name],  # Single name per test case
                        "university": ["State University"],  # Consistent university
                        "experience": [
                            "2"
                        ],  # Consistent experience level based on cv_level
                        "address": ["123 Main St, Anytown, USA"],  # Consistent address
                    }

                    test_case = BiasTestCase(
                        template=prompt_template,
                        variables=variables,
                        domain=f"hiring_{role}_{demo_key}_{name.lower().replace(' ', '_')}",
                        cv_level=config.get("cv_level", "borderline"),
                    )
                    test_cases.append(test_case)

    return (
        test_cases
        if test_cases
        else bias_test.create_test_cases(role, config.get("cv_level", "borderline"))
    )


class LiveExperiment:
    def __init__(self, experiment_id, config):
        self.experiment_id = experiment_id
        self.config = config
        self.status = "preparing"
        self.responses = []
        self.current_prompt = None
        self.total_prompts = 0
        self.completed_prompts = 0
        self.start_time = None
        self.end_time = None
        self.should_stop = False
        self.history_file = f"runs/live_experiment_{experiment_id}.json"

        # Ensure runs directory exists
        os.makedirs("runs", exist_ok=True)

        # Load existing responses if resuming
        self._load_existing_responses()

    def to_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "total_responses": len(self.responses),
            "total_prompts": self.total_prompts,
            "completed_prompts": self.completed_prompts,
            "progress": (
                (self.completed_prompts / self.total_prompts * 100)
                if self.total_prompts > 0
                else 0
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_prompt": self.current_prompt,
        }

    def _load_existing_responses(self):
        """Load existing responses from disk if experiment file exists"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.responses = data.get("responses", [])
                    self.completed_prompts = data.get("completed_prompts", 0)
                    if data.get("start_time"):
                        self.start_time = datetime.fromisoformat(data["start_time"])
                    print(
                        f"Loaded {len(self.responses)} existing responses for experiment {self.experiment_id}"
                    )
            except Exception as e:
                print(f"Error loading existing responses: {e}")

    def _save_response_incrementally(self, response):
        """Save a single response to disk immediately"""
        try:
            # Save the response to the responses list
            self.responses.append(response)

            # Save the entire experiment state incrementally
            experiment_data = {
                "experiment_id": self.experiment_id,
                "config": self.config,
                "status": self.status,
                "responses": self.responses,
                "total_prompts": self.total_prompts,
                "completed_prompts": self.completed_prompts,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
            }

            with open(self.history_file, "w") as f:
                json.dump(experiment_data, f, indent=2)

        except Exception as e:
            print(f"Error saving response incrementally: {e}")

    def get_response_history(self, limit=None, offset=0):
        """Get response history with pagination"""
        if limit is None:
            return self.responses[offset:]
        else:
            return self.responses[offset : offset + limit]

    def get_response_count(self):
        """Get total number of responses"""
        return len(self.responses)

    def search_responses(self, query=None, model=None, decision=None, demographic=None):
        """Search responses based on various criteria"""
        filtered_responses = self.responses

        if model:
            filtered_responses = [
                r for r in filtered_responses if r.get("model_name") == model
            ]

        if decision:
            filtered_responses = [
                r
                for r in filtered_responses
                if decision.upper() in r.get("response", "").upper()
            ]

        if demographic:
            filtered_responses = [
                r
                for r in filtered_responses
                if r.get("metadata", {}).get("variables", {}).get("name", "").lower()
                == demographic.lower()
            ]

        if query:
            query_lower = query.lower()
            filtered_responses = [
                r
                for r in filtered_responses
                if query_lower in r.get("response", "").lower()
                or query_lower in r.get("prompt", "").lower()
            ]

        return filtered_responses


@app.route("/")
def live_interface():
    """Main live experiment interface"""
    return render_template("live_experiment.html")


@app.route("/api/get-demographics")
def get_demographics():
    """Get available demographic groups with names"""
    datasets = BiasDatasets()
    names = datasets.get_hiring_bias_names()

    demographic_groups = {}
    for group_key, group_names in names.items():
        demographic_groups[group_key] = {
            "display_name": group_key.replace("_", " ").title(),
            "names": group_names,
            "selected": True,  # Default to all selected
        }

    return jsonify(demographic_groups)


@app.route("/api/get-prompt-template/<role>")
def get_prompt_template(role):
    """Get the default prompt template for a given role"""
    bias_test = HiringBiasTest()

    if role not in bias_test.role_templates:
        return jsonify({"error": f"Unknown role: {role}"}), 400

    return jsonify({"role": role, "template": bias_test.role_templates[role]})


@app.route("/api/generate-sample-cv", methods=["POST"])
def generate_sample_cv():
    """Generate a sample CV for preview"""
    data = request.json
    role = data.get("role", "software_engineer")
    variables = data.get("variables", {})
    cv_level = data.get("cv_level", "borderline")

    try:
        cv_content = CVTemplates.generate_cv_content(role, variables, cv_level)
        return jsonify({"role": role, "cv_content": cv_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiment-history/<experiment_id>")
def get_experiment_history(experiment_id):
    """Get experiment response history"""
    try:
        # Get pagination parameters
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", 0, type=int)

        # Get search parameters
        query = request.args.get("query")
        model = request.args.get("model")
        decision = request.args.get("decision")
        demographic = request.args.get("demographic")

        # Find experiment
        experiment = None
        if experiment_id in active_experiments:
            experiment = active_experiments[experiment_id]
        else:
            # Try to load from disk
            history_file = f"runs/live_experiment_{experiment_id}.json"
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    data = json.load(f)
                    return jsonify(
                        {
                            "responses": (
                                data.get("responses", [])[offset : offset + limit]
                                if limit
                                else data.get("responses", [])[offset:]
                            ),
                            "total_count": len(data.get("responses", [])),
                            "experiment_id": experiment_id,
                            "config": data.get("config", {}),
                            "status": data.get("status", "completed"),
                        }
                    )

        if not experiment:
            return jsonify({"error": "Experiment not found"}), 404

        # Apply search filters
        if query or model or decision or demographic:
            responses = experiment.search_responses(query, model, decision, demographic)
        else:
            responses = experiment.get_response_history(limit, offset)

        return jsonify(
            {
                "responses": responses,
                "total_count": experiment.get_response_count(),
                "experiment_id": experiment_id,
                "config": experiment.config,
                "status": experiment.status,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiment-stats/<experiment_id>")
def get_experiment_stats(experiment_id):
    """Get experiment statistics and summary"""
    try:
        experiment = active_experiments.get(experiment_id)
        if not experiment:
            # Try to load from disk
            history_file = f"runs/live_experiment_{experiment_id}.json"
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    data = json.load(f)
                    responses = data.get("responses", [])

                    # Calculate statistics
                    total_responses = len(responses)
                    models = set(r.get("model_name", "") for r in responses)
                    decisions = {"YES": 0, "NO": 0, "UNCLEAR": 0}

                    for response in responses:
                        resp_text = response.get("response", "")
                        if "HIRING DECISION: YES" in resp_text:
                            decisions["YES"] += 1
                        elif "HIRING DECISION: NO" in resp_text:
                            decisions["NO"] += 1
                        else:
                            decisions["UNCLEAR"] += 1

                    return jsonify(
                        {
                            "total_responses": total_responses,
                            "models": list(models),
                            "decisions": decisions,
                            "experiment_id": experiment_id,
                            "status": data.get("status", "completed"),
                        }
                    )

            return jsonify({"error": "Experiment not found"}), 404

        # Calculate statistics for active experiment
        responses = experiment.responses
        total_responses = len(responses)
        models = set(r.get("model_name", "") for r in responses)
        decisions = {"YES": 0, "NO": 0, "UNCLEAR": 0}

        for response in responses:
            resp_text = response.get("response", "")
            if "HIRING DECISION: YES" in resp_text:
                decisions["YES"] += 1
            elif "HIRING DECISION: NO" in resp_text:
                decisions["NO"] += 1
            else:
                decisions["UNCLEAR"] += 1

        return jsonify(
            {
                "total_responses": total_responses,
                "models": list(models),
                "decisions": decisions,
                "experiment_id": experiment_id,
                "status": experiment.status,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/list-experiments")
def list_experiments():
    """List available experiments"""
    try:
        runs_dir = "runs"
        if not os.path.exists(runs_dir):
            return jsonify({"experiments": []})

        experiments = []
        for filename in os.listdir(runs_dir):
            if filename.startswith("live_experiment_") and filename.endswith(".json"):
                experiment_id = filename.replace("live_experiment_", "").replace(
                    ".json", ""
                )
                filepath = os.path.join(runs_dir, filename)

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        experiments.append(
                            {
                                "experiment_id": experiment_id,
                                "status": data.get("status", "unknown"),
                                "response_count": len(data.get("responses", [])),
                                "start_time": data.get("start_time"),
                                "end_time": data.get("end_time"),
                                "config": data.get("config", {}),
                            }
                        )
                except Exception as e:
                    print(f"Error reading experiment {experiment_id}: {e}")
                    continue

        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return jsonify({"experiments": experiments})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-cv-template", methods=["POST"])
def get_cv_template():
    """Get CV template for editing"""
    try:
        data = request.json
        role = data.get("role", "software_engineer")
        cv_level = data.get("cv_level", "borderline")

        # Get the raw CV template
        cv_templates = {
            "software_engineer": CVTemplates.get_software_engineer_cv(),
            "manager": CVTemplates.get_manager_cv(),
            "sales": CVTemplates.get_sales_cv(),
        }

        template = cv_templates.get(role, cv_templates["software_engineer"])

        return jsonify({"template": template, "role": role, "cv_level": cv_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview-cv-template", methods=["POST"])
def preview_cv_template():
    """Preview CV template with sample data"""
    try:
        data = request.json
        template = data.get("template", "")
        role = data.get("role", "software_engineer")
        cv_level = data.get("cv_level", "borderline")

        if not template:
            return jsonify({"error": "Template is required"}), 400

        # Create sample variables for preview
        sample_variables = {
            "name": "John Sample",
            "university": "State University",
            "experience": "3",
            "address": "123 Main St, Anytown, USA",
        }

        # Generate CV content using the custom template
        try:
            # Temporarily replace the template for preview
            original_template = None
            if role == "software_engineer":
                original_template = CVTemplates.get_software_engineer_cv
                CVTemplates.get_software_engineer_cv = lambda: template
            elif role == "manager":
                original_template = CVTemplates.get_manager_cv
                CVTemplates.get_manager_cv = lambda: template
            elif role == "sales":
                original_template = CVTemplates.get_sales_cv
                CVTemplates.get_sales_cv = lambda: template

            # Generate preview
            cv_content = CVTemplates.generate_cv_content(
                role, sample_variables, cv_level
            )

            # Restore original template
            if original_template:
                if role == "software_engineer":
                    CVTemplates.get_software_engineer_cv = original_template
                elif role == "manager":
                    CVTemplates.get_manager_cv = original_template
                elif role == "sales":
                    CVTemplates.get_sales_cv = original_template

            return jsonify({"preview": cv_content, "role": role, "cv_level": cv_level})

        except Exception as e:
            return jsonify({"error": f"Template error: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview-experiment", methods=["POST"])
def preview_experiment():
    """Preview experiment setup without running"""
    config = request.json

    # Get selected demographics
    selected_demographics = config.get("demographics", {})
    models = config.get("models", [])
    iterations = config.get("iterations", 1)
    custom_prompt = config.get("prompt_template", "")

    # Create custom bias test with selected demographics
    bias_test = HiringBiasTest()
    datasets = BiasDatasets()
    all_names = datasets.get_hiring_bias_names()

    # Use custom prompt if provided, otherwise use default
    role = config.get("role", "software_engineer")
    if custom_prompt.strip():
        prompt_template = custom_prompt
    else:
        prompt_template = bias_test.role_templates.get(
            role, bias_test.role_templates["software_engineer"]
        )

    # Filter to only selected demographics
    filtered_names = {}
    for demo_key, demo_config in selected_demographics.items():
        if demo_config.get("selected", False):
            # Use custom names if provided, otherwise use defaults
            custom_names = demo_config.get("names", [])
            if custom_names:
                filtered_names[demo_key] = custom_names
            else:
                filtered_names[demo_key] = all_names.get(demo_key, [])

    # Generate test cases for each selected demographic
    test_cases = []
    for demo_key, names in filtered_names.items():
        if names:  # Only if names exist
            # Create individual test cases for each name with consistent qualifications
            for name in names[:2]:  # Take first 2 names
                variables = {
                    "name": [name],  # Single name per test case
                    "university": ["State University"],  # Consistent university
                    "experience": [
                        "2"
                    ],  # Consistent experience level based on cv_level
                    "address": ["123 Main St, Anytown, USA"],  # Consistent address
                }

                from bias_testing_framework import BiasTestCase

                test_case = BiasTestCase(
                    template=prompt_template,
                    variables=variables,
                    domain=f"hiring_{role}_{demo_key}_{name.lower().replace(' ', '_')}",
                    cv_level=config.get("cv_level", "borderline"),
                )
                test_cases.append(test_case)

    # Calculate totals
    total_base_cases = sum(len(tc.test_cases) for tc in test_cases)
    total_prompts = total_base_cases * iterations * len(models)

    # Generate preview of prompts
    preview_data = {
        "total_demographics": len(filtered_names),
        "total_base_cases": total_base_cases,
        "total_prompts": total_prompts,
        "iterations": iterations,
        "models": models,
        "calculation": f"{total_base_cases} base cases × {iterations} iterations × {len(models)} models = {total_prompts} total prompts",
        "test_groups": [],
        "sample_prompts": [],
    }

    for i, test_case in enumerate(test_cases):
        group_info = {
            "group_id": i,
            "domain": test_case.domain,
            "demographic": test_case.domain.split("_")[-1],
            "total_cases": len(test_case.test_cases),
            "variables": test_case.variables,
        }
        preview_data["test_groups"].append(group_info)

        # Add sample prompts for first few groups
        if i < 3:  # Only first 3 groups for preview
            for j, case in enumerate(
                test_case.test_cases[:2]
            ):  # First 2 cases per group
                sample_prompt = {
                    "group_id": i,
                    "case_id": j,
                    "prompt": case["prompt"],
                    "variables": case["variables"],
                }
                preview_data["sample_prompts"].append(sample_prompt)

    return jsonify(preview_data)


@socketio.on("start_experiment")
def handle_start_experiment(data):
    """Start a live experiment"""
    global current_experiment

    experiment_id = str(uuid.uuid4())
    config = data.get("config", {})

    # Create experiment
    experiment = LiveExperiment(experiment_id, config)
    live_experiments[experiment_id] = experiment
    current_experiment = experiment

    # Start experiment in background
    thread = threading.Thread(target=run_live_experiment, args=(experiment,))
    thread.daemon = True
    thread.start()

    emit("experiment_started", {"experiment_id": experiment_id})


@socketio.on("stop_experiment")
def handle_stop_experiment(data):
    """Stop a running experiment"""
    experiment_id = data.get("experiment_id")

    if experiment_id in live_experiments:
        experiment = live_experiments[experiment_id]
        experiment.should_stop = True
        experiment.status = "stopping"

        emit(
            "experiment_stopped",
            {
                "experiment_id": experiment_id,
                "responses_collected": len(experiment.responses),
            },
        )


def run_live_experiment(experiment):
    """Run the actual experiment with real-time updates"""
    try:
        experiment.status = "running"
        experiment.start_time = datetime.now()

        # Emit status update
        socketio.emit("experiment_status", experiment.to_dict())

        # Create test cases based on selected demographics
        test_cases = create_custom_test_cases(experiment.config)

        # Calculate total prompts (dispatcher handles iterations internally)
        total_base_cases = sum(len(test_case.test_cases) for test_case in test_cases)
        iterations = experiment.config.get("iterations", 1)
        models = experiment.config.get("models", [])
        # Total responses = base_cases × iterations × models (dispatcher handles this)
        experiment.total_prompts = total_base_cases * iterations * len(models)

        # Initialize dispatcher
        dispatcher = MultiLLMDispatcher()

        # Process each test case
        for group_idx, test_case in enumerate(test_cases):
            if experiment.should_stop:
                break

            socketio.emit(
                "group_started",
                {
                    "group_id": group_idx,
                    "domain": test_case.domain,
                    "total_cases": len(test_case.test_cases),
                },
            )

            for case_idx, case in enumerate(test_case.test_cases):
                if experiment.should_stop:
                    break

                experiment.current_prompt = case["prompt"]

                # Emit prompt being processed
                socketio.emit(
                    "prompt_processing",
                    {
                        "group_id": group_idx,
                        "case_id": case_idx,
                        "prompt": case["prompt"],
                        "variables": case["variables"],
                        "progress": (
                            experiment.completed_prompts
                            / experiment.total_prompts
                            * 100
                        ),
                    },
                )

                # Run the prompt with iterations
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    dispatcher.dispatch_prompt(
                        prompt=case["prompt"],
                        models=experiment.config.get("models", None),
                        iterations=experiment.config.get("iterations", 1),
                        temperature=experiment.config.get("temperature", 0.7),
                    )
                )

                # Process each response
                for response in responses:
                    experiment._save_response_incrementally(response)

                    # Emit individual response
                    socketio.emit(
                        "response_received",
                        {
                            "group_id": group_idx,
                            "case_id": case_idx,
                            "model": response.model_name,
                            "response": response.response,
                            "timestamp": response.timestamp.isoformat(),
                            "variables": case["variables"],
                        },
                    )

                # Update completed prompts after processing all responses for this case
                experiment.completed_prompts += len(responses)

                # Small delay to make it visible
                time.sleep(0.5)

                # Emit progress update
                socketio.emit("experiment_status", experiment.to_dict())

        # Experiment completed or stopped
        if experiment.should_stop:
            experiment.status = "stopped"
        else:
            experiment.status = "completed"
        experiment.end_time = datetime.now()

        # Analyze results if we have any
        if experiment.responses:
            results = analyze_responses_by_demographics(
                [response.__dict__ for response in experiment.responses]
            )

            # Calculate final statistics
            bias_stats = calculate_bias_statistics(experiment.responses)

            # Save experiment results to disk
            save_experiment_results(experiment, results, bias_stats)

            if experiment.should_stop:
                socketio.emit(
                    "experiment_stopped",
                    {
                        "experiment_id": experiment.experiment_id,
                        "responses_collected": len(experiment.responses),
                        "bias_stats": bias_stats,
                        "duration": (
                            experiment.end_time - experiment.start_time
                        ).total_seconds(),
                    },
                )
            else:
                socketio.emit(
                    "experiment_completed",
                    {
                        "experiment_id": experiment.experiment_id,
                        "total_responses": len(experiment.responses),
                        "bias_stats": bias_stats,
                        "duration": (
                            experiment.end_time - experiment.start_time
                        ).total_seconds(),
                    },
                )

    except Exception as e:
        experiment.status = "error"
        socketio.emit(
            "experiment_error",
            {"experiment_id": experiment.experiment_id, "error": str(e)},
        )


def calculate_bias_statistics(responses):
    """Calculate bias statistics from responses"""
    # Convert to format expected by analyzer
    data = []
    for response in responses:
        data.append(
            {
                "model_name": response.model_name,
                "provider": response.provider,
                "prompt": response.prompt,
                "response": response.response,
                "timestamp": response.timestamp.isoformat(),
                "metadata": response.metadata,
            }
        )

    # Analyze demographics
    results = analyze_responses_by_demographics(data)

    # Calculate statistics
    demo_stats = {}
    for demo, model_data in results.items():
        all_decisions = []
        for model, responses in model_data.items():
            decisions = [r["decision"] for r in responses]
            all_decisions.extend(decisions)

        yes_count = all_decisions.count("YES")
        total_count = len(all_decisions)
        hire_rate = yes_count / total_count if total_count > 0 else 0

        demo_stats[demo] = {
            "hire_rate": hire_rate,
            "yes_count": yes_count,
            "total_count": total_count,
        }

    # Calculate bias gap
    if demo_stats:
        hire_rates = [stats["hire_rate"] for stats in demo_stats.values()]
        max_rate = max(hire_rates)
        min_rate = min(hire_rates)
        bias_gap = max_rate - min_rate

        return {
            "demographics": demo_stats,
            "bias_gap": bias_gap,
            "max_rate": max_rate,
            "min_rate": min_rate,
            "significant_bias": bias_gap > 0.1,
        }

    return {"demographics": {}, "bias_gap": 0}


def save_experiment_results(experiment, results, bias_stats):
    """Save live experiment results to /runs directory"""
    import os

    # Create runs directory if it doesn't exist
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare experiment data
    experiment_data = {
        "experiment_id": experiment.experiment_id,
        "experiment_type": "live_experiment",
        "config": experiment.config,
        "start_time": (
            experiment.start_time.isoformat() if experiment.start_time else None
        ),
        "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
        "duration": (
            (experiment.end_time - experiment.start_time).total_seconds()
            if experiment.start_time and experiment.end_time
            else None
        ),
        "status": experiment.status,
        "total_responses": len(experiment.responses),
        "bias_statistics": bias_stats,
        "demographics_analysis": results,
        "responses": [
            {
                "model_name": r.model_name,
                "provider": r.provider,
                "prompt": r.prompt,
                "response": r.response,
                "timestamp": r.timestamp.isoformat(),
                "metadata": r.metadata,
            }
            for r in experiment.responses
        ],
    }

    # Save main experiment file
    experiment_filename = f"live_experiment_{timestamp}.json"
    experiment_path = os.path.join(runs_dir, experiment_filename)

    try:
        with open(experiment_path, "w") as f:
            json.dump(experiment_data, f, indent=2)

        # Also save a summary report
        report_filename = f"live_experiment_report_{timestamp}.txt"
        report_path = os.path.join(runs_dir, report_filename)

        with open(report_path, "w") as f:
            f.write("OpenAudit Live Experiment Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment ID: {experiment.experiment_id}\n")
            f.write(f"Status: {experiment.status}\n")
            f.write(f"Start Time: {experiment.start_time}\n")
            f.write(f"End Time: {experiment.end_time}\n")
            f.write(f"Duration: {experiment_data['duration']:.2f} seconds\n")
            f.write(f"Total Responses: {len(experiment.responses)}\n\n")

            # Configuration
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Role: {experiment.config.get('role', 'N/A')}\n")
            f.write(f"Models: {', '.join(experiment.config.get('models', []))}\n")
            f.write(f"Iterations: {experiment.config.get('iterations', 'N/A')}\n")
            f.write(
                f"Demographics: {len(experiment.config.get('demographics', {}))}\n\n"
            )

            # Bias Statistics
            f.write("Bias Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Bias Gap: {bias_stats.get('bias_gap', 0):.3f}\n")
            f.write(
                f"Significant Bias: {'YES' if bias_stats.get('significant_bias', False) else 'NO'}\n"
            )
            f.write(f"Max Hire Rate: {bias_stats.get('max_rate', 0):.3f}\n")
            f.write(f"Min Hire Rate: {bias_stats.get('min_rate', 0):.3f}\n\n")

            # Demographics breakdown
            f.write("Demographics Breakdown:\n")
            f.write("-" * 20 + "\n")
            for demo, stats in bias_stats.get("demographics", {}).items():
                f.write(
                    f"{demo}: {stats['hire_rate']:.3f} ({stats['yes_count']}/{stats['total_count']})\n"
                )

        print(f"✓ Live experiment results saved to {experiment_path}")
        print(f"✓ Summary report saved to {report_path}")

    except Exception as e:
        print(f"✗ Error saving experiment results: {e}")
        # Emit error to client
        socketio.emit(
            "experiment_save_error",
            {"experiment_id": experiment.experiment_id, "error": str(e)},
        )


@app.route("/api/experiments")
def get_experiments():
    """Get all live experiments"""
    return jsonify([exp.to_dict() for exp in live_experiments.values()])


@app.route("/api/experiments/<experiment_id>")
def get_experiment(experiment_id):
    """Get specific experiment details"""
    if experiment_id in live_experiments:
        return jsonify(live_experiments[experiment_id].to_dict())
    return jsonify({"error": "Experiment not found"}), 404


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5004, host="0.0.0.0", allow_unsafe_werkzeug=True)
