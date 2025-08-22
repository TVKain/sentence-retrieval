"""
Plot scripts for sentence retrieval JSON artifacts

Take in one or more JSON artifacts
x-axis: layer
y-axis: accuracy
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(file_path: str):
    """Load a JSON artifact."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_layer_accuracy(json_files: list[str], save_folder: str):
    plt.figure(figsize=(10, 5))  # wider figure
    legend_labels = []

    for jf in json_files:
        data = load_json(jf)
        layer_acc_pairs = data.get("layer_accuracy", [])
        layer_acc_pairs.sort(key=lambda x: x[0])
        layers = [pair[0] for pair in layer_acc_pairs]
        acc = [pair[1] for pair in layer_acc_pairs]

        model_name = Path(data["model_path"]).parent.name
        checkpoint = Path(data["model_path"]).name
        base_lang = data.get("base_lang", "unknown")
        target_lang = data.get("target_lang", "unknown")

        label = f"{model_name}/{checkpoint} ({base_lang}-{target_lang})"
        legend_labels.append(label)

        plt.plot(layers, acc, marker="o", label=label)

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Sentence Retrieval Accuracy per Layer")
    plt.grid(True)

    # Legend outside the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)  # leave space for legend

    # Save plot and metadata
    Path(save_folder).mkdir(exist_ok=True, parents=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_filename = f"{timestamp_str}_plot.png"
    metadata_filename = f"{timestamp_str}_plot.json"

    plot_path = Path(save_folder) / plot_filename
    plt.savefig(plot_path, bbox_inches='tight')  # ensures legend is not clipped
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Save metadata JSON
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "plot_file": str(plot_path),
        "json_sources": json_files,
        "base_target_pairs": [
            {"base_lang": load_json(jf).get("base_lang", "unknown"),
             "target_lang": load_json(jf).get("target_lang", "unknown")}
            for jf in json_files
        ]
    }

    metadata_path = Path(save_folder) / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Plot metadata saved to {metadata_path}")

    return plot_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Sentence retrieval Plot")
    parser.add_argument(
        "--json", nargs="+", required=True, help="Path(s) to JSON artifact(s)"
    )
    parser.add_argument(
        "--plot-artifact-out", type=str, default="plot_artifacts", help="Path to save the plot (optional)"
    )

    args = parser.parse_args()

    plot_layer_accuracy(args.json, save_folder=args.plot_artifact_out)


if __name__ == "__main__":
    main()
