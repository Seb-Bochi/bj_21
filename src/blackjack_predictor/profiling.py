import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from blackjack_predictor.data_.dataset import BlackjackDataset
from blackjack_predictor.models.ffnn import SimpleFNN


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def run_profiling(cfg: DictConfig) -> None:
    data_path = cfg.profiling_config.data_path
    num_batches = cfg.profiling_config.num_batches
    batch_size = cfg.profiling_config.batch_size
    row_limit = cfg.profiling_config.row_limit

    model = SimpleFNN()
    model.eval()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        # ── 1. Dataset init (reads & parses the CSV) ──────────────────
        with record_function("dataset_init"):
            dataset = BlackjackDataset(
                processed_file=data_path
            )  # <-- your dataset class that loads the processed .pt file

        # ── 2. DataLoader construction ────────────────────────────────
        with record_function("dataloader_init"):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # ── 3. Batch fetching (tensor conversion + collation) ─────────
        batches = []
        with record_function("data_loading"):
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                batches.append(batch)

        # ── 4. Model inference ────────────────────────────────────────
        with torch.no_grad():
            for x, _ in batches:
                with record_function("model_inference"):
                    model(x)

    # ── Full table sorted by total CPU time ───────────────────────────
    print("\n=== FULL PROFILE (sorted by CPU total) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit))

    # ── Focused summary: only your named sections ─────────────────────
    print("\n=== PIPELINE STAGE SUMMARY ===")
    named = {
        e.key: e
        for e in prof.key_averages()
        if e.key
        in {
            "dataset_init",
            "dataloader_init",
            "data_loading",
            "model_inference",
        }
    }
    stages = ["dataset_init", "dataloader_init", "data_loading", "model_inference"]
    total = sum(named[s].cpu_time_total for s in stages if s in named)

    print(f"{'Stage':<25} {'Total (ms)':>12} {'Avg (ms)':>10} {'Calls':>7} {'% of pipeline':>15}")
    print("-" * 72)
    for stage in stages:
        if stage not in named:
            continue
        e = named[stage]
        total_ms = e.cpu_time_total / 1000
        avg_ms = e.cpu_time_total / max(e.count, 1) / 1000
        pct = 100 * e.cpu_time_total / total if total else 0
        print(f"{stage:<25} {total_ms:>12.3f} {avg_ms:>10.3f} {e.count:>7} {pct:>14.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<25} {total/1000:>12.3f}")


if __name__ == "__main__":
    run_profiling()
