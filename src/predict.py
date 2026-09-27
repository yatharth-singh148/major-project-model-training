import torch
import random

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "./models/muril_combined"

INDIC_DATASET_NAME = "maitry30/Multilingual_Fake_News_Detection"

TRUE_FILE = "./data/True.csv"
FAKE_FILE = "./data/Fake.csv"

MAX_LENGTH = 192


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# Load model
# ==========================================

print("\nLoading trained MuRIL model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ==========================================
# Labels
# ==========================================

LABELS = {
    0: "REAL",
    1: "FAKE"
}


# ==========================================
# Prediction function
# ==========================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()

    return LABELS[prediction], confidence


# ==========================================
# Load random examples
# ==========================================

def load_random_examples():

    print("\nLoading random examples...")

    indic_dataset = load_dataset(
        INDIC_DATASET_NAME,
        split="train"
    )

    examples = []

    # Add Indic examples

    for _ in range(5):

        index = random.randint(
            0,
            len(indic_dataset) - 1
        )

        row = indic_dataset[index]

        examples.append({
            "text": row["clean_text"],
            "label": int(row["label"]),
            "language": row["language"]
        })


    # Add English examples

    import pandas as pd

    true_df = pd.read_csv(TRUE_FILE)
    fake_df = pd.read_csv(FAKE_FILE)

    # Pick random English samples

    for _ in range(5):

        if random.choice([True, False]):

            row = true_df.sample(
                1
            ).iloc[0]

            label = 0

        else:

            row = fake_df.sample(
                1
            ).iloc[0]

            label = 1

        title = str(row["title"])
        text = str(row["text"])

        combined_text = (
            "Title: "
            + title
            + "\nArticle: "
            + text
        )

        examples.append({
            "text": combined_text,
            "label": label,
            "language": "English"
        })

    return examples


# ==========================================
# Custom prediction
# ==========================================

def custom_prediction():

    print("\n===================================")
    print("Enter News Article")
    print("===================================\n")

    print("Paste the news text below.")
    print("Press ENTER twice when finished.\n")

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:

        print("\nNo text entered.")

        return

    prediction, confidence = predict(
        text
    )

    print("\n===================================")
    print("Prediction")
    print("===================================")

    print(
        f"Result: {prediction}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("===================================\n")


# ==========================================
# Random prediction
# ==========================================

def random_prediction():

    examples = load_random_examples()

    example = random.choice(
        examples
    )

    text = example["text"]
    actual_label = LABELS[
        example["label"]
    ]

    prediction, confidence = predict(
        text
    )

    print("\n===================================")
    print("Random News Example")
    print("===================================\n")

    print(
        "Language:",
        example["language"]
    )

    print("\nNews:")
    print("-----------------------------------")

    # Don't flood the console with huge articles

    print(
        text[:1500]
    )

    if len(text) > 1500:
        print("...")

    print("-----------------------------------")

    print(
        "\nActual:",
        actual_label
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    if prediction == actual_label:

        print(
            "\nCorrect prediction."
        )

    else:

        print(
            "\nIncorrect prediction."
        )

    print("===================================\n")


# ==========================================
# Main menu
# ==========================================

while True:

    print("\n===================================")
    print("     MuRIL Fake News Detector")
    print("===================================")

    print("1. Enter your own news")
    print("2. Pick a random test example")
    print("3. Exit")

    choice = input(
        "\nChoose an option: "
    ).strip()

    if choice == "1":

        custom_prediction()

    elif choice == "2":

        random_prediction()

    elif choice == "3":

        print("\nExiting...")

        break

    else:

        print(
            "\nInvalid choice. Please select 1, 2 or 3."
        )