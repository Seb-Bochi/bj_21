import time
from pathlib import Path

import torch
import torch.nn as nn

from blackjack_predictor.models.ffnn import SimpleFNN


def generate_mock_eval_data(num_samples=1000, input_dim=13):
    X = torch.rand(num_samples, input_dim, dtype=torch.float32)
    y = (X.sum(dim=1) > (input_dim / 2)).long()
    return X, y


def evaluate_model(model, X, y):
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    latency = (time.perf_counter() - start_time) / len(X) * 1000
    return accuracy, latency


def run_optimization_and_drift_pipeline():
    print("=== Starting Model Optimization & Robustness Pipeline ===")

    input_dim = 13
    model = SimpleFNN(input_dim=input_dim, hidden_dim=128, output_dim=2)
    X, y = generate_mock_eval_data()

    results = {}

    base_acc, base_lat = evaluate_model(model, X, y)
    results["Baseline"] = {"Accuracy": base_acc, "Latency_ms_per_sample": base_lat}

    print("Running torch.compile()...")
    try:
        compiled_model = torch.compile(model)
        _ = compiled_model(X[:5])
        comp_acc, comp_lat = evaluate_model(compiled_model, X, y)
        results["Torch_Compile"] = {"Accuracy": comp_acc, "Latency_ms_per_sample": comp_lat}
    except Exception as e:
        print(f"torch.compile skipped or unsupported on this platform: {e}")
        results["Torch_Compile"] = {"Accuracy": "N/A", "Latency_ms_per_sample": "N/A"}

    print("Applying Dynamic Quantization...")
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quant_acc, quant_lat = evaluate_model(quantized_model, X, y)
    results["Quantized_int8"] = {"Accuracy": quant_acc, "Latency_ms_per_sample": quant_lat}

    quant_path = Path("models/quantized_int8_model.pt")
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), quant_path)

    print("Exporting model to ONNX format...")
    onnx_path = "models/optimized_deployment_model.onnx"
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    results["ONNX_Export"] = {"Status": "Exported Successfully", "Path": onnx_path}

    print("Running Synthetic Data Drift Robustness Experiment...")
    drift_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    drift_robustness_results = {}

    for level in drift_levels:
        noise = torch.randn_like(X) * level
        drifted_X = X + noise

        drift_acc, _ = evaluate_model(model, drifted_X, y)
        drift_robustness_results[f"Noise_Std_{level}"] = round(drift_acc, 4)

    results["Synthetic_Drift_Robustness_Experiment"] = drift_robustness_results

    report_path = Path("reports/optimization_and_drift_results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Model Optimization & Data Drift Robustness Report\n\n")
        f.write("## 1. Optimization & Latency Benchmark Results\n")
        f.write("| Strategy | Accuracy / Status | Latency (ms/sample) |\n")
        f.write("| --- | --- | --- |\n")
        f.write(
            f"| Baseline | {results['Baseline']['Accuracy']:.4f} | {results['Baseline']['Latency_ms_per_sample']:.6f} |\n"
        )
        f.write(
            f"| torch.compile | {results['Torch_Compile']['Accuracy']} | {results['Torch_Compile']['Latency_ms_per_sample']} |\n"
        )
        f.write(
            f"| Quantized (int8) | {results['Quantized_int8']['Accuracy']:.4f} | {results['Quantized_int8']['Latency_ms_per_sample']:.6f} |\n"
        )
        f.write(f"| ONNX Export | {results['ONNX_Export']['Status']} | N/A |\n\n")

        f.write("## 2. Synthetic Data Drift Robustness Evaluation\n")
        f.write("This section measures how model performance degrades under synthetic feature drift stress levels.\n\n")
        f.write("| Noise Deviation Level | Model Evaluation Accuracy |\n")
        f.write("| --- | --- |\n")
        for k, v in drift_robustness_results.items():
            f.write(f"| {k} | {v} |\n")

    print(f"Successfully wrote concrete compliance performance results to {report_path}")


if __name__ == "__main__":
    run_optimization_and_drift_pipeline()
