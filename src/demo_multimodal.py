import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, ViTModel

MURIL_PATH = "./models/muril_combined"
VIT_NAME = "WinKawaks/vit-tiny-patch16-224"
MAX_LENGTH = 192

LABELS = {0: "REAL", 1: "FAKE"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("     MULTIMODAL FAKE NEWS DETECTION DEMO")
print("=" * 55)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading MuRIL...")
tokenizer = AutoTokenizer.from_pretrained(MURIL_PATH)
muril = AutoModelForSequenceClassification.from_pretrained(MURIL_PATH)
muril.to(device)
muril.eval()

print("Loading ViT...")
image_processor = AutoImageProcessor.from_pretrained(VIT_NAME)
vit = ViTModel.from_pretrained(VIT_NAME)

# ViT is only a feature extractor in this demo.
for param in vit.parameters():
    param.requires_grad = False

vit.to(device)
vit.eval()

print("Models loaded successfully.")


def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = muril(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, prediction].item()

    return LABELS[prediction], confidence


def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vit(**inputs)
        image_features = outputs.last_hidden_state[:, 0, :]

    return image_features


while True:
    print("\n" + "=" * 55)
    print("1. Analyze text + image")
    print("2. Analyze text only")
    print("3. Exit")
    print("=" * 55)

    choice = input("Choose an option: ").strip()

    if choice == "1":
        print("\nEnter the news text.")
        text = input("> ").strip()

        if not text:
            print("No text entered.")
            continue

        image_path = input("Enter image path: ").strip()

        try:
            prediction, confidence = predict_text(text)
            image_features = extract_image_features(image_path)

            print("\n" + "-" * 55)
            print("MULTIMODAL ANALYSIS")
            print("-" * 55)

            print("\nTEXT ANALYSIS")
            print("Prediction :", prediction)
            print(f"Confidence : {confidence * 100:.2f}%")

            print("\nIMAGE ANALYSIS")
            print("ViT model   :", VIT_NAME)
            print("Feature size:", image_features.shape[-1])
            print("Status      : Image successfully encoded")

            print("\nPIPELINE")
            print("MuRIL -> Text representation")
            print("ViT   -> Visual representation")
            print("       -> Fusion layer (next training stage)")
            print("       -> Final multimodal REAL/FAKE decision")

            print("\nNOTE:")
            print("This demo shows both encoders working.")
            print("The final multimodal classifier will make the")
            print("combined decision after fusion is trained.")

        except Exception as e:
            print("\nCould not process the image.")
            print("Error:", e)

    elif choice == "2":
        print("\nEnter the news text.")
        text = input("> ").strip()

        if not text:
            print("No text entered.")
            continue

        prediction, confidence = predict_text(text)

        print("\n" + "-" * 55)
        print("TEXT ANALYSIS")
        print("-" * 55)
        print("Prediction :", prediction)
        print(f"Confidence : {confidence * 100:.2f}%")

    elif choice == "3":
        print("\nExiting demo.")
        break

    else:
        print("\nInvalid choice.")
