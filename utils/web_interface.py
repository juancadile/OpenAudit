"""
OpenAudit Web Interface
Simple Flask app to visualize bias testing results
"""

import glob
import json
import os
from collections import defaultdict
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from core.response_analyzer import analyze_responses_by_demographics

app = Flask(__name__)


def get_available_runs():
    """Get all available experiment runs"""
    runs = []
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")

    print(f"DEBUG: Looking for runs in: {runs_dir}")
    print(f"DEBUG: Runs directory exists: {os.path.exists(runs_dir)}")

    if os.path.exists(runs_dir):
        # Look for both hiring_bias_experiment and live_experiment files
        pattern1 = os.path.join(runs_dir, "hiring_bias_experiment_*.json")
        pattern2 = os.path.join(runs_dir, "live_experiment_*.json")
        print(f"DEBUG: Using patterns: {pattern1}, {pattern2}")
        files = glob.glob(pattern1) + glob.glob(pattern2)
        print(f"DEBUG: Found files: {files}")
    else:
        files = []

    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract timestamp from filename
        # Format: hiring_bias_experiment_20250715_223226.json or live_experiment_20250716_151903.json
        parts = filename.replace(".json", "").split("_")

        # Handle both file patterns
        if filename.startswith("hiring_bias_experiment_") and len(parts) >= 4:
            date_part = parts[-2]  # 20250715
            time_part = parts[-1]  # 223226
            timestamp_str = f"{date_part}_{time_part}"
            experiment_type = "Hiring Bias"
        elif filename.startswith("live_experiment_") and len(parts) >= 3:
            # Check if the last part is a timestamp or UUID
            if len(parts[-1]) == 6 and parts[-1].isdigit():  # HHMMSS format
                date_part = parts[-2]  # 20250716
                time_part = parts[-1]  # 151903
                timestamp_str = f"{date_part}_{time_part}"
                experiment_type = "Live Experiment"
            else:
                # This is a UUID format, skip it for now
                print(f"DEBUG: Skipping UUID format file: {filename}")
                continue
        else:
            print(f"DEBUG: Filename format not recognized: {filename}")
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
        except ValueError as e:
            print(f"DEBUG: Failed to parse timestamp '{timestamp_str}': {e}")
            continue

    return sorted(runs, key=lambda x: x["timestamp_raw"], reverse=True)


def load_and_analyze_run(filepath):
    """Load and analyze a specific run"""
    # Make sure we have the full path
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)

    print(f"DEBUG: Loading file: {filepath}")
    print(f"DEBUG: File exists: {os.path.exists(filepath)}")

    with open(filepath, "r") as f:
        raw_data = json.load(f)

    # Handle both data formats
    if isinstance(raw_data, dict) and "responses" in raw_data:
        # Live experiment format: extract responses array
        print("DEBUG: Detected live experiment format")
        data = raw_data["responses"]
    elif isinstance(raw_data, list):
        # Original CLI format: data is already the responses array
        print("DEBUG: Detected CLI experiment format")
        data = raw_data
    else:
        raise ValueError(f"Unknown data format in file {filepath}")

    # Basic stats
    total_responses = len(data)
    models = list(set(item["model_name"] for item in data))

    # Analyze by demographics
    demographic_results = analyze_responses_by_demographics(data)

    # Format for web display
    analysis = {
        "total_responses": total_responses,
        "models": models,
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
    for model in models:
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

        # Find which demographics have max/min rates
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

    return analysis


@app.route("/")
def index():
    """Main dashboard page"""
    runs = get_available_runs()
    print(f"DEBUG: Found {len(runs)} runs: {[r['filename'] for r in runs]}")
    return render_template("index.html", runs=runs)


@app.route("/api/runs")
def api_runs():
    """API endpoint to get available runs"""
    return jsonify(get_available_runs())


@app.route("/api/analyze/<run_id>")
def api_analyze(run_id):
    """API endpoint to analyze a specific run"""
    runs = get_available_runs()
    run = next((r for r in runs if r["timestamp_raw"] == run_id), None)

    if not run:
        return jsonify({"error": "Run not found"}), 404

    try:
        analysis = load_and_analyze_run(run["filepath"])
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
                print(f"DEBUG: Unknown data format in file {run['filepath']}, skipping")
                continue

            aggregated_data.extend(data)
        except Exception as e:
            print(f"DEBUG: Error loading {run['filepath']}: {e}")
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
        for demo, model_data in demographic_results.items():
            if model in model_data:
                decisions = [r["decision"] for r in model_data[model]]
                model_decisions.extend(decisions)

        overall_yes = model_decisions.count("YES")
        overall_total = len(model_decisions)
        overall_rate = overall_yes / overall_total if overall_total > 0 else 0

        analysis["models_analysis"][model] = {
            "overall_hire_rate": overall_rate,
            "overall_hire_rate_percent": f"{overall_rate*100:.1f}%",
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


if __name__ == "__main__":
    app.run(debug=True, port=5002)
