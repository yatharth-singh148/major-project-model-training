from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATASET_NAME = "maitry30/Multilingual_Fake_News_Detection"
NUM_SAMPLES = 5000
RANDOM_STATE = 42


print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)["train"]

# Use the same 5,000 samples as our MuRIL experiment
dataset = dataset.shuffle(seed=RANDOM_STATE).select(
    range(min(NUM_SAMPLES, len(dataset)))
)

texts = dataset["clean_text"]
labels = dataset["label"]


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

indices = list(range(len(dataset)))

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


train_texts = [texts[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

validation_texts = [texts[i] for i in validation_indices]
validation_labels = [labels[i] for i in validation_indices]

test_texts = [texts[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]


print("\nDataset split:")
print(f"Train: {len(train_texts)}")
print(f"Validation: {len(validation_texts)}")
print(f"Test: {len(test_texts)}")


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_texts)
X_validation = vectorizer.transform(validation_texts)
X_test = vectorizer.transform(test_texts)

print(f"Training feature shape: {X_train.shape}")


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

model.fit(X_train, train_labels)


# --------------------------------------------------
# Validation
# --------------------------------------------------

validation_predictions = model.predict(X_validation)

print("\nValidation Results:")
print(f"Accuracy : {accuracy_score(validation_labels, validation_predictions):.4f}")
print(f"Precision: {precision_score(validation_labels, validation_predictions):.4f}")
print(f"Recall   : {recall_score(validation_labels, validation_predictions):.4f}")
print(f"F1-score : {f1_score(validation_labels, validation_predictions):.4f}")


# --------------------------------------------------
# Test
# --------------------------------------------------

test_predictions = model.predict(X_test)

print("\n===================================")
print("Final Test Results")
print("===================================")

print(f"Accuracy : {accuracy_score(test_labels, test_predictions):.4f}")
print(f"Precision: {precision_score(test_labels, test_predictions):.4f}")
print(f"Recall   : {recall_score(test_labels, test_predictions):.4f}")
print(f"F1-score : {f1_score(test_labels, test_predictions):.4f}")