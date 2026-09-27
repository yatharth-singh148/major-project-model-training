import os
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    ViTModel,
)

# ============================================================
# MuRIL + ViT MULTIMODAL FAKE NEWS DETECTOR
# ============================================================
#
# FIRST VERSION:
# - MuRIL encoder: pretrained weights from our trained text model
# - ViT-Tiny encoder: pretrained ImageNet weights
# - Both encoders are FROZEN initially
# - Their embeddings are projected and concatenated
# - A small fusion classifier learns REAL/FAKE
#
# This is intentionally lightweight for the RTX 4050 6 GB.
#
# DATA:
# data/mmifnd.csv
#
# Required CSV columns:
#   text        -> news article text
#   image_path  -> path to corresponding image
#   label       -> MMIFND label: 0 = fake, 1 = real
#   language    -> optional language name
#
# We convert labels to our project convention:
#   0 = REAL
#   1 = FAKE
# ============================================================

MURIL_PATH = "./models/muril_combined"
VIT_NAME = "WinKawaks/vit-tiny-patch16-224"

DATA_FILE = "./data/mmifnd.csv"
OUTPUT_DIR = "./models/muril_vit"

MAX_TEXT_LENGTH = 192
NUM_SAMPLES = 6000
RANDOM_STATE = 42

BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3

TEXT_DIM = 768
IMAGE_DIM = 192
FUSION_DIM = 256

LABELS = {
    0: "REAL",
    1: "FAKE"
}


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("       MuRIL + ViT MULTIMODAL TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

class MultimodalNewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, image_processor):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        text = str(row["text"])
        image_path = str(row["image_path"])

        image = Image.open(image_path).convert("RGB")

        text_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"
        )

        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )

        item = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(
                int(row["project_label"]),
                dtype=torch.long
            )
        }

        return item


# ============================================================
# Load manifest
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nCould not find {DATA_FILE}\n\n"
        "Create the MMIFND manifest first with columns:\n"
        "text,image_path,label,language\n"
        "where MMIFND label is 0=fake and 1=real."
    )

print("\nLoading multimodal dataset...")
df = pd.read_csv(DATA_FILE)

required_columns = ["text", "image_path", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

# Convert MMIFND convention:
# 0 = fake, 1 = real
#
# Our project convention:
# 0 = real, 1 = fake
df["project_label"] = 1 - df["label"].astype(int)

# Remove missing entries.
df = df.dropna(
    subset=["text", "image_path", "label"]
).copy()

# Keep only rows whose images actually exist.
df = df[
    df["image_path"].apply(
        lambda path: os.path.exists(str(path))
    )
].reset_index(drop=True)

print("Usable samples:", len(df))

if len(df) == 0:
    raise ValueError(
        "No usable samples found. Check image_path values."
    )

# Small first experiment for the laptop.
df = df.sample(
    n=min(NUM_SAMPLES, len(df)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("Samples used:", len(df))


# ============================================================
# Train / validation / test split
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["project_label"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["project_label"]
)

print("\nDataset split:")
print("Train      :", len(train_df))
print("Validation :", len(validation_df))
print("Test       :", len(test_df))


# ============================================================
# Load encoders
# ============================================================

print("\nLoading MuRIL...")

tokenizer = AutoTokenizer.from_pretrained(
    MURIL_PATH
)

# Load the BERT backbone from our already fine-tuned
# MuRIL sequence-classification checkpoint.
text_encoder = AutoModel.from_pretrained(
    MURIL_PATH
)

text_encoder.to(device)
text_encoder.eval()


print("Loading ViT-Tiny...")

image_processor = AutoImageProcessor.from_pretrained(
    VIT_NAME
)

image_encoder = ViTModel.from_pretrained(
    VIT_NAME
)

image_encoder.to(device)
image_encoder.eval()


# ============================================================
# Freeze both encoders for the first multimodal experiment
# ============================================================

for parameter in text_encoder.parameters():
    parameter.requires_grad = False

for parameter in image_encoder.parameters():
    parameter.requires_grad = False


# ============================================================
# Fusion model
# ============================================================

class MultimodalFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(TEXT_DIM, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.image_projection = nn.Sequential(
            nn.Linear(IMAGE_DIM, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2)
        )

    def forward(self, text_features, image_features):

        text_features = self.text_projection(
            text_features
        )

        image_features = self.image_projection(
            image_features
        )

        fused = torch.cat(
            [text_features, image_features],
            dim=1
        )

        return self.classifier(fused)


fusion_model = MultimodalFusion().to(device)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = MultimodalNewsDataset(
    train_df,
    tokenizer,
    image_processor
)

validation_dataset = MultimodalNewsDataset(
    validation_df,
    tokenizer,
    image_processor
)

test_dataset = MultimodalNewsDataset(
    test_df,
    tokenizer,
    image_processor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Optimizer / loss
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    fusion_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# Feature extraction
# ============================================================

def get_features(batch):

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    with torch.no_grad():

        text_outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # MuRIL [CLS] representation
        text_features = text_outputs.last_hidden_state[:, 0, :]

        image_outputs = image_encoder(
            pixel_values=pixel_values
        )

        # ViT [CLS] representation
        image_features = image_outputs.last_hidden_state[:, 0, :]

    return text_features, image_features


# ============================================================
# Evaluation
# ============================================================

def evaluate(loader):

    fusion_model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            labels = batch["label"].to(device)

            text_features, image_features = get_features(
                batch
            )

            logits = fusion_model(
                text_features,
                image_features
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    )

    return (
        total_loss / max(len(loader), 1),
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("Starting multimodal training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    fusion_model.train()

    running_loss = 0.0

    for step, batch in enumerate(train_loader):

        labels = batch["label"].to(device)

        text_features, image_features = get_features(
            batch
        )

        logits = fusion_model(
            text_features,
            image_features
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 50 == 0:

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Step {step + 1}/{len(train_loader)} | "
                f"Loss: {running_loss / (step + 1):.4f}"
            )

    val_loss, val_acc, val_precision, val_recall, val_f1 = (
        evaluate(validation_loader)
    )

    print("\nValidation:")
    print(f"Accuracy : {val_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall   : {val_recall:.4f}")
    print(f"F1       : {val_f1:.4f}")


# ============================================================
# Final test
# ============================================================

print("\n" + "=" * 60)
print("Final Multimodal Test Results")
print("=" * 60)

test_loss, test_acc, test_precision, test_recall, test_f1 = (
    evaluate(test_loader)
)

print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")


# ============================================================
# Save fusion model
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.save(
    fusion_model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "fusion_model.pt"
    )
)

print("\nMultimodal fusion model saved to:")
print(OUTPUT_DIR)

print("\nArchitecture:")
print("MuRIL (frozen) -> 768-dim text features")
print("ViT-Tiny (frozen) -> 192-dim image features")
print("        ↓")
print("Projection + Concatenation")
print("        ↓")
print("MLP classifier")
print("        ↓")
print("REAL / FAKE")
