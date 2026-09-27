from datasets import load_dataset
from collections import Counter

DATASET_NAME = "maitry30/Multilingual_Fake_News_Detection"

dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"]

print("\n===== DATASET =====")
print(dataset)

print("\n===== COLUMNS =====")
print(train_data.column_names)

print("\n===== NUMBER OF SAMPLES =====")
print(len(train_data))

print("\n===== LABEL DISTRIBUTION =====")
print(Counter(train_data["label"]))

print("\n===== LANGUAGE DISTRIBUTION =====")
print(Counter(train_data["language"]))

print("\n===== LABEL 0 SAMPLE =====")
for sample in train_data:
    if sample["label"] == 0:
        print(sample)
        break

print("\n===== LABEL 1 SAMPLE =====")
for sample in train_data:
    if sample["label"] == 1:
        print(sample)
        break