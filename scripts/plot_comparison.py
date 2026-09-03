"""Generate comparison graph and benchmark figures from comparison.json."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    comp_json_path = root / "data" / "runs" / "pruned_seed_0" / "comparison.json"
    if not comp_json_path.exists():
        print(f"File not found: {comp_json_path}")
        return

    data = json.loads(comp_json_path.read_text())
    ours = data["ours"]

    categories = ["clean", "windy", "dns_synthetic", "speech_noise"]
    cat_names = ["Clean Speech", "Windy (Reverb)", "DNS Synthetic", "Speech + Noise"]

    models = [
        ("PulseVAD Teacher (81k)", ours["teacher_81k"], "#2563eb"),
        ("PulseVAD Ship INT8 (2.1k)", ours["student_2.1k_int8"], "#059669"),
        ("Silero-VAD v6 (measured)", ours["Silero-VAD v6 (measured, same windows)"], "#dc2626"),
        ("Silero-VAD v5 (measured)", ours["Silero-VAD v5 (measured, same windows)"], "#ea580c"),
        ("MarbleNet (91k, measured)", ours["MarbleNet (measured, same windows)"], "#7c3aed"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 1. Measured AUC across 4 acoustic categories
    x = np.arange(len(categories))
    width = 0.16
    for idx, (mname, row, color) in enumerate(models):
        aucs = [row[c]["auc"] for c in categories]
        rects = axes[0].bar(x + (idx - 2.0) * width, aucs, width, label=mname, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
        for r in rects:
            h = r.get_height()
            axes[0].annotate(f"{h:.3f}",
                             xy=(r.get_x() + r.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=6.8, rotation=45)

    axes[0].set_ylabel("AUC-ROC (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0].set_title("Measured AUC Across Held-Out Categories\n(Exact Same 200 ms Windows, Causal)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cat_names, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0.75, 1.03)
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Pareto Frontier: Model Size (Params, Log Scale) vs Clean & Windy AUC
    pareto_models = [
        ("PulseVAD 2.1k INT8", 2118, 0.976, 0.938, "#059669", "*", 220),
        ("PulseVAD 81k Teacher", 81090, 0.989, 0.985, "#2563eb", "s", 110),
        ("Silero-VAD v6 (measured)", 309000, 0.988, 0.934, "#dc2626", "^", 120),
        ("Silero-VAD v5 (measured)", 545000, 0.990, 0.960, "#ea580c", "v", 110),
        ("MarbleNet (measured)", 91000, 0.970, 0.910, "#7c3aed", "d", 110),
        ("TinyVAD (cited, non-causal)", 11600, 0.950, 0.864, "#d97706", "x", 80),
        ("ResectNet (cited)", 4500, 0.940, 0.886, "#0891b2", "p", 80),
        ("AtomicVAD (cited)", 300, 0.920, 0.869, "#4b5563", "X", 80),
    ]

    for name, params, clean_auc, windy_auc, color, marker, size in pareto_models:
        axes[1].scatter([params], [windy_auc], color=color, marker=marker, s=size, label=name, edgecolors="black", linewidth=0.8, zorder=5)
        offset_y = 0.005 if "Silero" not in name else -0.015
        offset_x = 1.15 if params < 10000 else 0.4
        axes[1].annotate(name.split(" ")[0], xy=(params, windy_auc), xytext=(params * offset_x, windy_auc + offset_y),
                         fontsize=8.5, fontweight="semibold")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parameter Count (Log Scale, Lower is Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Windy / Noise AUC (Robustness)", fontsize=11, fontweight="bold")
    axes[1].set_title("Size vs Robustness Trade-off\n(PulseVAD 2.1k vs Competitors)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.82, 1.01)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_img = root / "data" / "runs" / "pruned_seed_0" / "comparison_graph.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"Comparison graph saved to: {out_img}")

    # 3. Multilingual Benchmark Graph
    multi_json = root / "data" / "runs" / "pruned_seed_0" / "multilingual_report.json"
    if multi_json.exists():
        mdata = json.loads(multi_json.read_text())["report"]
        langs = list(mdata.keys())
        pulse_i8 = [mdata[l]["PulseVAD Ship INT8 (2.1k)"]["auc"] for l in langs]
        pulse_tea = [mdata[l]["PulseVAD Teacher (81k)"]["auc"] for l in langs]
        silero_v5 = [mdata[l]["Silero-VAD v5 (545k)"]["auc"] for l in langs]

        # Append Macro Average
        langs.append("Macro Average")
        pulse_i8.append(float(np.mean(pulse_i8)))
        pulse_tea.append(float(np.mean(pulse_tea)))
        silero_v5.append(float(np.mean(silero_v5)))

        fig_m, ax_m = plt.subplots(figsize=(14, 5.5), dpi=300)
        xm = np.arange(len(langs))
        w = 0.26
        r1 = ax_m.bar(xm - w, pulse_i8, w, label="PulseVAD 2.1k INT8", color="#059669", alpha=0.9, edgecolor="black", linewidth=0.5)
        r2 = ax_m.bar(xm, pulse_tea, w, label="PulseVAD 81k Teacher", color="#2563eb", alpha=0.9, edgecolor="black", linewidth=0.5)
        r3 = ax_m.bar(xm + w, silero_v5, w, label="Silero-VAD v5 (545k)", color="#dc2626", alpha=0.9, edgecolor="black", linewidth=0.5)

        for rects in [r1, r2, r3]:
            for r in rects:
                h = r.get_height()
                ax_m.annotate(f"{h:.2f}",
                              xy=(r.get_x() + r.get_width() / 2, h),
                              xytext=(0, 2), textcoords="offset points",
                              ha='center', va='bottom', fontsize=7, rotation=45)

        ax_m.set_ylabel("AUC-ROC (Higher is Better)", fontsize=11, fontweight="bold")
        ax_m.set_title("Multilingual Benchmark Across 10 Languages (FLEURS + Noise)\nIncluding Indian Languages (Hindi, Tamil, Telugu, Bengali)", fontsize=12, fontweight="bold")
        ax_m.set_xticks(xm)
        ax_m.set_xticklabels(langs, fontsize=9.5, fontweight="bold", rotation=25, ha="right")
        ax_m.set_ylim(0.65, 1.02)
        ax_m.axvline(x=len(langs) - 1.5, color="gray", linestyle="--", alpha=0.7, label="_nolegend_")
        ax_m.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
        ax_m.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_m_img = root / "data" / "runs" / "pruned_seed_0" / "multilingual_graph.png"
        plt.savefig(out_m_img, dpi=300, bbox_inches="tight")
        print(f"Multilingual graph saved to: {out_m_img}")

if __name__ == "__main__":
    main()
