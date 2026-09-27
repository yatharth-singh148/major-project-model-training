"""
Generate MODEL_METRICS.txt from the currently reported experiments.

This script does NOT train or evaluate a model.
It only records the current experimental metrics in a clean report.

Run:
    python generate_model_metrics.py
"""

from pathlib import Path
from datetime import datetime

MODELS = [
    {
        "name": "MuRIL - Combined Indic + English",
        "type": "Transformer text classifier",
        "checkpoint": "./models/muril_combined",
        "validation": {
            "Accuracy": 0.9984,
            "Precision": 0.9996,
            "Recall": 0.9976,
            "F1-score": 0.9986,
        },
        "test": {
            "Accuracy": 0.9976,
            "Precision": 0.9987,
            "Recall": 0.9970,
            "F1-score": 0.9979,
        },
    },
    {
        "name": "TF-IDF + Logistic Regression",
        "type": "Traditional ML baseline",
        "checkpoint": "N/A",
        "validation": None,
        "test": {
            "Accuracy": 0.9900,
            "Precision": 0.9871,
            "Recall": 0.9968,
            "F1-score": 0.9919,
        },
    },
]


def format_metrics(metrics):
    if metrics is None:
        return "Not available"

    return "\n".join(
        f"    {name:<10}: {value:.4f} ({value * 100:.2f}%)"
        for name, value in metrics.items()
    )


def main():
    output = Path("MODEL_METRICS.txt")

    lines = [
        "MULTIMODAL FAKE NEWS DETECTION",
        "CURRENT MODEL METRICS",
        "=============================",
        "",
        f"Report generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "NOTE",
        "----",
        "These are dataset/split-specific experimental results.",
        "They should not be interpreted as universal real-world accuracy.",
        "",
    ]

    for number, model in enumerate(MODELS, 1):
        lines += [
            f"{number}. {model['name']}",
            "-" * (len(model["name"]) + 3),
            f"Type:       {model['type']}",
            f"Checkpoint: {model['checkpoint']}",
            "",
            "Validation metrics:",
            format_metrics(model["validation"]),
            "",
            "Test metrics:",
            format_metrics(model["test"]),
            "",
        ]

    lines += [
        "CURRENT MULTIMODAL STATUS",
        "--------------------------",
        "Image encoder: planned/pretrained encoder not yet evaluated",
        "Multimodal fusion: not yet trained/evaluated",
        "Real-time verification: backend component planned",
        "",
        "INTERPRETATION",
        "--------------",
        "The MuRIL model is a text classifier. Its confidence is not equivalent",
        "to factual certainty. Real-time evidence verification is intended to",
        "provide a separate factual-support signal.",
        "",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {output.resolve()}")


if __name__ == "__main__":
    main()
