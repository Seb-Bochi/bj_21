import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

def generate_drift_report(reference_csv: str, current_csv: str, output_path: str = "report.html"):
    # 1. Load reference (e.g., your baseline training data) and current production inputs
    ref_data = pd.read_csv(reference_csv)
    curr_data = pd.read_csv(current_csv)

    # 2. Build an Evidently report tracking Data, Target, and Quality metrics
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset()
    ])

    # 3. Calculate drift and export to an interactive HTML dashboard
    report.run(reference_data=ref_data, current_data=curr_data)
    report.save_html(output_path)
    print(f"Drift report generated and saved to {output_path}")

if __name__ == "__main__":
    # Test it with a mock current file to check your model's sensitivity
    generate_drift_report("data/raw/blkjckhands.csv", "data/raw/blkjckhands.csv")