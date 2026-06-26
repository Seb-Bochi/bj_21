import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report


def generate_drift_report(reference_csv: str, current_csv: str, output_path: str = "report.html"):
    """Generate an Evidently drift report from two CSV files."""

    ref_data = pd.read_csv(reference_csv)
    curr_data = pd.read_csv(current_csv)

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset()])

    report.run(reference_data=ref_data, current_data=curr_data)
    report.save_html(output_path)
    print(f"Drift report generated and saved to {output_path}")


if __name__ == "__main__":
    generate_drift_report("data/raw/blkjckhands.csv", "data/raw/blkjckhands.csv")
