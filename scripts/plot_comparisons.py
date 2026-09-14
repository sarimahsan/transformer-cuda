import os
import json
import argparse
import numpy as np


def generate_plots(results_dir: str = "results", output_fig_dir: str = "results/figures"):
    os.makedirs(output_fig_dir, exist_ok=True)

    pt_path = os.path.join(results_dir, "pytorch_benchmark.json")
    cu_path = os.path.join(results_dir, "cuda_benchmark.json")
    gla_path = os.path.join(results_dir, "gla_comparison.json")
    conv_path = os.path.join(results_dir, "convergence_comparison.json")
    scale_path = os.path.join(results_dir, "scaling_comparison.json")

    has_pt = os.path.exists(pt_path)
    has_cu = os.path.exists(cu_path)
    has_gla = os.path.exists(gla_path)
    has_conv = os.path.exists(conv_path)
    has_scale = os.path.exists(scale_path)

    data = {}
    if has_pt:
        with open(pt_path, "r") as f:
            data.update(json.load(f))
    if has_cu:
        with open(cu_path, "r") as f:
            data["cuda"] = json.load(f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------
        # Plot 1: Full Step Latency Baseline
        # ------------------------------------------------------------
        if data:
            frameworks = [v.get("framework", k) for k, v in data.items()]
            step_latencies = [v["step"]["mean_ms"] for v in data.values()]
            step_stds = [v["step"].get("std_ms", 0.0) for v in data.values()]

            plt.figure(figsize=(9, 5), dpi=150)
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
            bars = plt.bar(frameworks, step_latencies, yerr=step_stds, capsize=5, color=colors[:len(frameworks)], edgecolor="black", alpha=0.85)
            plt.ylabel("Full Step Latency (ms) [Lower is Better]", fontsize=11, fontweight="bold")
            plt.title("Transformer Full Step Training Latency (B=32, T=256, L=6, d=256)", fontsize=13, fontweight="bold")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            for bar, lat in zip(bars, step_latencies):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{lat:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

            plt.tight_layout()
            p1 = os.path.join(output_fig_dir, "benchmark_comparison.png")
            plt.savefig(p1)
            plt.close()

            # Plot 2: Phase Breakdown
            fwd = [v["forward"]["mean_ms"] for v in data.values()]
            bwd = [v["backward"]["mean_ms"] for v in data.values()]
            opt = [v["optimizer"]["mean_ms"] for v in data.values()]

            plt.figure(figsize=(9, 5), dpi=150)
            x_indices = np.arange(len(frameworks))
            width = 0.55

            plt.bar(x_indices, fwd, width, label="Forward Pass", color="#3498db", edgecolor="black")
            plt.bar(x_indices, bwd, width, bottom=fwd, label="Backward Pass", color="#e74c3c", edgecolor="black")
            plt.bar(x_indices, opt, width, bottom=np.array(fwd) + np.array(bwd), label="AdamW Optimizer", color="#2ecc71", edgecolor="black")

            plt.ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
            plt.title("Phase Breakdown: Forward vs. Backward vs. AdamW Step", fontsize=13, fontweight="bold")
            plt.xticks(x_indices, frameworks)
            plt.legend(loc="upper right", framealpha=0.9)
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            p2 = os.path.join(output_fig_dir, "breakdown_phases.png")
            plt.savefig(p2)
            plt.close()

        # ------------------------------------------------------------
        # Plot 3: Architectural Throughput Comparison
        # ------------------------------------------------------------
        if has_gla:
            with open(gla_path, "r") as f:
                gla_data = json.load(f)

            labels = []
            throughputs = []
            bar_colors = []
            
            mapping = [
                ("gpt_eager", "GPT-2 (Eager)", "#7f8c8d"),
                ("gpt_compile", "GPT-2 (torch.compile)", "#e67e22"),
                ("gla_eager", "FastTransformer (Eager)", "#2980b9"),
                ("gla_compile", "FastTransformer (compile)", "#27ae60")
            ]

            for key, label, col in mapping:
                if key in gla_data and gla_data[key] is not None:
                    labels.append(label)
                    throughputs.append(gla_data[key]["tokens_per_sec"])
                    bar_colors.append(col)

            plt.figure(figsize=(10, 5), dpi=150)
            bars = plt.bar(labels, throughputs, color=bar_colors, edgecolor="black", alpha=0.9, width=0.55)
            plt.ylabel("Throughput (Tokens / Sec) [Higher is Better]", fontsize=11, fontweight="bold")
            plt.title("Throughput Breakthrough: Standard GPT-2 vs. FastTransformer (Tesla T4)", fontsize=13, fontweight="bold")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            for bar, tps in zip(bars, throughputs):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500, f"{tps:,.0f} tok/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

            plt.tight_layout()
            p3 = os.path.join(output_fig_dir, "architecture_throughput.png")
            plt.savefig(p3)
            plt.close()
            print(f"[Plot] Generated architectural throughput plot: {p3}")

        # ------------------------------------------------------------
        # Plot 4: Convergence & Perplexity Curves
        # ------------------------------------------------------------
        if has_conv:
            with open(conv_path, "r") as f:
                conv_data = json.load(f)

            gpt_loss = conv_data["gpt2"]["losses"]
            fast_loss = conv_data["fast_transformer"]["losses"]
            steps = conv_data["gpt2"]["steps"]

            plt.figure(figsize=(9, 5), dpi=150)
            plt.plot(steps, gpt_loss, label=f"Standard GPT-2 (Final Loss: {conv_data['gpt2']['final_loss']:.4f}, PPL: {conv_data['gpt2']['final_ppl']:.2f})", color="#e74c3c", linewidth=2.2, marker="o", markersize=4)
            plt.plot(steps, fast_loss, label=f"FastTransformer (Final Loss: {conv_data['fast_transformer']['final_loss']:.4f}, PPL: {conv_data['fast_transformer']['final_ppl']:.2f})", color="#2ecc71", linewidth=2.2, marker="s", markersize=4)
            plt.xlabel("Training Steps (TinyShakespeare)", fontsize=11, fontweight="bold")
            plt.ylabel("Cross-Entropy Loss [Lower is Better]", fontsize=11, fontweight="bold")
            plt.title("Convergence Parity: Standard GPT-2 vs. FastTransformer (200 Steps)", fontsize=13, fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right", framealpha=0.9)

            plt.tight_layout()
            p4 = os.path.join(output_fig_dir, "convergence_curve.png")
            plt.savefig(p4)
            plt.close()
            print(f"[Plot] Generated convergence curve: {p4}")

        # ------------------------------------------------------------
        # Plot 5: Sequence Length Scaling Curve
        # ------------------------------------------------------------
        if has_scale:
            with open(scale_path, "r") as f:
                scale_data = json.load(f)

            lengths = scale_data["lengths"]
            gpt_tok = [v["tokens_per_sec"] if v else 0 for v in scale_data["gpt_compile"]]
            fast_tok = [v["tokens_per_sec"] if v else 0 for v in scale_data["fast_compile"]]

            plt.figure(figsize=(9, 5), dpi=150)
            plt.plot(lengths, gpt_tok, label="Standard GPT-2 (compile)", color="#e67e22", marker="o", linewidth=2)
            plt.plot(lengths, fast_tok, label="FastTransformer (compile)", color="#27ae60", marker="s", linewidth=2)
            plt.xlabel("Sequence Length (T)", fontsize=11, fontweight="bold")
            plt.ylabel("Token Throughput (tok/s)", fontsize=11, fontweight="bold")
            plt.title("Sequence Length Scaling: Throughput vs. Context Window", fontsize=13, fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right", framealpha=0.9)

            plt.tight_layout()
            p5 = os.path.join(output_fig_dir, "sequence_scaling.png")
            plt.savefig(p5)
            plt.close()
            print(f"[Plot] Generated sequence scaling curve: {p5}")

    except ImportError:
        print("[Plot] matplotlib not installed. Skipping graphical plot generation.")


if __name__ == "__main__":
    generate_plots()
