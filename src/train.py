import os
import torch
import pandas as pd

from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
    Value
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "google/muril-base-cased"

INDIC_DATASET_NAME = "maitry30/Multilingual_Fake_News_Detection"

TRUE_FILE = "./data/True.csv"
FAKE_FILE = "./data/Fake.csv"

OUTPUT_DIR = "./models/muril_combined"

MAX_LENGTH = 192

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 1. Load Indic dataset
# ==========================================

print("\n===================================")
print("Loading Indic dataset")
print("===================================\n")

indic_dataset = load_dataset(
    INDIC_DATASET_NAME,
    split="train"
)

print("Indic samples:", len(indic_dataset))

# Keep only the columns we need
indic_dataset = indic_dataset.select_columns(
    ["clean_text", "label", "language"]
)

# Rename clean_text -> text
indic_dataset = indic_dataset.rename_column(
    "clean_text",
    "text"
)


# ==========================================
# 2. Load English datasets
# ==========================================

print("\n===================================")
print("Loading English datasets")
print("===================================\n")

true_df = pd.read_csv(TRUE_FILE)
fake_df = pd.read_csv(FAKE_FILE)

print("True samples:", len(true_df))
print("Fake samples:", len(fake_df))


# ==========================================
# 3. Prepare English data
# ==========================================

# True news = 0
true_df["label"] = 0

# Fake news = 1
fake_df["label"] = 1

true_df["language"] = "English"
fake_df["language"] = "English"


# ==========================================
# 4. Combine title + article text
# ==========================================

def create_text(df):

    df = df.copy()

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["combined_text"] = (
        "Title: "
        + df["title"].astype(str)
        + "\nArticle: "
        + df["text"].astype(str)
    )

    return df


true_df = create_text(true_df)
fake_df = create_text(fake_df)


# Keep only required columns

true_df = true_df[
    ["combined_text", "label", "language"]
]

fake_df = fake_df[
    ["combined_text", "label", "language"]
]


# Rename combined_text -> text

true_df = true_df.rename(
    columns={"combined_text": "text"}
)

fake_df = fake_df.rename(
    columns={"combined_text": "text"}
)


# Convert pandas -> Hugging Face Dataset

true_dataset = Dataset.from_pandas(
    true_df,
    preserve_index=False
)

fake_dataset = Dataset.from_pandas(
    fake_df,
    preserve_index=False
)

# Make English schema match the Indic dataset
true_dataset = true_dataset.cast_column(
    "text",
    Value("string")
)

fake_dataset = fake_dataset.cast_column(
    "text",
    Value("string")
)

true_dataset = true_dataset.cast_column(
    "language",
    Value("string")
)

fake_dataset = fake_dataset.cast_column(
    "language",
    Value("string")
)


# ==========================================
# 5. Combine English datasets
# ==========================================

english_dataset = concatenate_datasets(
    [
        true_dataset,
        fake_dataset
    ]
)

print(
    "English samples:",
    len(english_dataset)
)


# ==========================================
# 6. Combine English + Indic
# ==========================================

dataset = concatenate_datasets(
    [
        indic_dataset,
        english_dataset
    ]
)

print("\n===================================")
print("Combined dataset")
print("===================================\n")

print("Total samples:", len(dataset))


# ==========================================
# 7. Shuffle dataset
# ==========================================

dataset = dataset.shuffle(
    seed=42
)


# ==========================================
# 8. Train / validation / test split
# ==========================================

print("\nCreating train / validation / test split...")

labels = dataset["label"]

indices = list(
    range(len(dataset))
)


# 80% train
# 20% temporary

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# Labels for temporary set

temp_labels = [
    labels[i]
    for i in temp_indices
]


# 10% validation
# 10% test

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


# Create datasets

train_dataset = dataset.select(
    train_indices
)

validation_dataset = dataset.select(
    validation_indices
)

test_dataset = dataset.select(
    test_indices
)


print("\nDataset split:")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(validation_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ==========================================
# 9. Load tokenizer
# ==========================================

print("\n===================================")
print("Loading MuRIL tokenizer")
print("===================================\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# 10. Tokenization
# ==========================================

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    desc="Tokenizing train"
)

print("Tokenizing validation dataset...")

validation_dataset = validation_dataset.map(
    tokenize,
    batched=True,
    desc="Tokenizing validation"
)

print("Tokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize,
    batched=True,
    desc="Tokenizing test"
)


# ==========================================
# 11. Prepare dataset format
# ==========================================

columns = [
    "input_ids",
    "attention_mask",
    "label"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

validation_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)


# ==========================================
# 12. Load MuRIL
# ==========================================

print("\n===================================")
print("Loading MuRIL model")
print("===================================\n")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ==========================================
# 13. Evaluation metrics
# ==========================================

def compute_metrics(pred):

    predictions = pred.predictions.argmax(axis=1)

    labels = pred.label_ids

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# 14. Training configuration
# ==========================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,

    num_train_epochs=1,

    weight_decay=0.01,

    fp16=True,

    logging_steps=100,

    load_best_model_at_end=False,

    report_to="none"
)


# ==========================================
# 15. Trainer
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics,

    data_collator=data_collator
)


# ==========================================
# 16. Train
# ==========================================

print("\n===================================")
print("Starting MuRIL training")
print("===================================\n")

trainer.train()


# ==========================================
# 17. Final evaluation
# ==========================================

print("\n===================================")
print("Evaluating on test set")
print("===================================\n")

results = trainer.evaluate(
    test_dataset
)


print("\nFinal Results:")

for key, value in results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ==========================================
# 18. Save model
# ==========================================

print("\n===================================")
print("Saving model")
print("===================================\n")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"\nModel saved to {OUTPUT_DIR}"
)


# ==========================================
# 19. Sample prediction
# ==========================================

print("\n===================================")
print("Sample Prediction")
print("===================================\n")


# Take one real sample from test set

# Get the raw sample from the original combined dataset
sample = dataset.select(
    [test_indices[0]]
)[0]

sample_text = sample["text"]

actual_label = sample["label"]

# Tokenize sample

inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH
)


# Move inputs to GPU if available

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# Prediction

model.eval()

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


# Convert labels to readable text

label_names = {
    0: "REAL",
    1: "FAKE"
}


print("Input:")
print("-----------------------------------")

print(
    sample_text[:1000]
)

if len(sample_text) > 1000:
    print("...")


print("\n-----------------------------------")

print(
    "Actual:",
    label_names[actual_label]
)

print(
    "Prediction:",
    label_names[prediction]
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print("-----------------------------------")