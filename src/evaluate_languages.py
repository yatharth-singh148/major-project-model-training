from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd


DATASET_NAME = "maitry30/Multilingual_Fake_News_Detection"
MODEL_PATH = "./models/muril"

NUM_SAMPLES = 5000
RANDOM_STATE = 42
MAX_LENGTH = 256


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)["train"]

dataset = dataset.shuffle(seed=RANDOM_STATE).select(
    range(min(NUM_SAMPLES, len(dataset)))
)

labels = dataset["label"]
indices = list(range(len(dataset)))


# --------------------------------------------------
# Reproduce original split
# --------------------------------------------------

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels
)

temp_labels = [labels[i] for i in temp_indices]

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_labels
)

test_dataset = dataset.select(test_indices)

print(f"Test samples: {len(test_dataset)}")


# --------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------

print("\nLoading MuRIL model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# --------------------------------------------------
# Tokenize test data
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["clean_text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing test data...")

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

trainer = Trainer(
    model=model,
    data_collator=data_collator
)

print("\nRunning predictions...")

predictions = trainer.predict(tokenized_test)

predicted_labels = predictions.predictions.argmax(axis=-1)

true_labels = test_dataset["label"]


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

print("\n===================================")
print("Overall Test Results")
print("===================================")

print(
    f"Accuracy : "
    f"{accuracy_score(true_labels, predicted_labels):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(true_labels, predicted_labels):.4f}"
)

print(
    f"Recall   : "
    f"{recall_score(true_labels, predicted_labels):.4f}"
)

print(
    f"F1-score : "
    f"{f1_score(true_labels, predicted_labels):.4f}"
)


# --------------------------------------------------
# Language-wise metrics
# --------------------------------------------------

print("\n===================================")
print("Language-wise Results")
print("===================================")

languages = sorted(set(test_dataset["language"]))

results = []

for language in languages:

    language_indices = [
        i for i, lang in enumerate(test_dataset["language"])
        if lang == language
    ]

    language_true = [
        true_labels[i]
        for i in language_indices
    ]

    language_pred = [
        predicted_labels[i]
        for i in language_indices
    ]

    results.append({
        "Language": language,
        "Samples": len(language_indices),
        "Accuracy": accuracy_score(
            language_true,
            language_pred
        ),
        "Precision": precision_score(
            language_true,
            language_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            language_true,
            language_pred,
            zero_division=0
        ),
        "F1": f1_score(
            language_true,
            language_pred,
            zero_division=0
        )
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)