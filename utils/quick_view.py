#!/usr/bin/env python3
"""
Quick viewer for OpenAudit results - works without Flask
"""

import glob
import json
import os
from datetime import datetime

from core.response_analyzer import analyze_responses_by_demographics


def get_runs():
    """Get all experiment runs"""
    runs = []
    files = glob.glob("runs/hiring_bias_experiment_*.json")

    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 4:
            date_part = parts[-2]  # 20250715
            time_part = parts[-1]  # 223226
            timestamp_str = f"{date_part}_{time_part}"
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                runs.append(
                    {
                        "filepath": filepath,
                        "filename": filename,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp_raw": timestamp_str,
                    }
                )
            except ValueError:
                continue

    return sorted(runs, key=lambda x: x["timestamp_raw"], reverse=True)


def analyze_run(filepath):
    """Analyze a single run"""
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    # Basic stats
    total_responses = len(data)
    models = list(set(item["model_name"] for item in data))

    print(f"Total Responses: {total_responses}")
    print(f"Models: {', '.join(models)}")

    # Analyze demographics
    results = analyze_responses_by_demographics(data)

    print(f"\nHIRING RATES BY DEMOGRAPHIC:")
    print("-" * 40)

    demo_rates = {}
    for demo, model_data in results.items():
        all_decisions = []
        for model, responses in model_data.items():
            decisions = [r["decision"] for r in responses]
            all_decisions.extend(decisions)

        yes_count = all_decisions.count("YES")
        total_count = len(all_decisions)
        hire_rate = yes_count / total_count if total_count > 0 else 0
        demo_rates[demo] = hire_rate

        print(
            f"{demo.replace('_', ' ').title():20} {hire_rate:6.1%} ({yes_count}/{total_count})"
        )

    # Bias analysis
    if demo_rates:
        max_rate = max(demo_rates.values())
        min_rate = min(demo_rates.values())
        bias_gap = max_rate - min_rate

        max_demo = max(demo_rates, key=demo_rates.get)
        min_demo = min(demo_rates, key=demo_rates.get)

        print(f"\nBIAS ANALYSIS:")
        print("-" * 40)
        print(f"Highest Rate: {max_demo.replace('_', ' ').title()} ({max_rate:.1%})")
        print(f"Lowest Rate:  {min_demo.replace('_', ' ').title()} ({min_rate:.1%})")
        print(f"Bias Gap:     {bias_gap:.1%}")

        if bias_gap > 0.1:
            print("🚨 SIGNIFICANT BIAS DETECTED (>10% gap)")
        elif bias_gap > 0.05:
            print("⚠️  MODERATE BIAS DETECTED (>5% gap)")
        else:
            print("✅ No significant bias detected (<5% gap)")

    # Model consistency
    print(f"\nMODEL CONSISTENCY:")
    print("-" * 40)

    for model in models:
        model_decisions = []
        for demo, model_data in results.items():
            if model in model_data:
                decisions = [r["decision"] for r in model_data[model]]
                model_decisions.extend(decisions)

        if model_decisions:
            yes_count = model_decisions.count("YES")
            total_count = len(model_decisions)
            rate = yes_count / total_count
            print(f"{model:20} {rate:6.1%} ({yes_count}/{total_count})")


def main():
    """Main function"""
    runs = get_runs()

    if not runs:
        print("No experiment runs found in runs/ directory.")
        print("Run bias_testing_framework.py to generate data.")
        return

    print("OpenAudit Quick Results Viewer")
    print("=" * 60)

    print(f"\nFound {len(runs)} experiment runs:")
    for i, run in enumerate(runs):
        print(f"{i+1}. {run['timestamp']} - {run['filename']}")

    print(f"\nAnalyzing most recent run: {runs[0]['filename']}")
    analyze_run(runs[0]["filepath"])

    print(f"\n{'='*60}")
    print("For web interface, run: python3 web_interface.py")
    print("Then visit: http://localhost:5002")


if __name__ == "__main__":
    main()
