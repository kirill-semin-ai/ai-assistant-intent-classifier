from pathlib import Path
import sys

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "intent_classifier.joblib"


def load_model(path: Path):
    """Load the trained intent classification model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run training first: python src/train.py"
        )

    return joblib.load(path)


def predict_intent(text: str) -> None:
    """Predict intent for a single user message."""
    model = load_model(MODEL_PATH)

    prediction = model.predict([text])[0]

    print("AI Assistant Intent Classifier")
    print("=" * 40)
    print(f"Input text: {text}")
    print(f"Predicted intent: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        print()
        print("Class probabilities:")
        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        for class_name, probability in ranked:
            print(f"- {class_name}: {probability:.3f}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python src/predict.py "find information about neural networks"')
        sys.exit(1)

    text = " ".join(sys.argv[1:]).strip()

    if not text:
        print("Error: input text is empty.")
        sys.exit(1)

    predict_intent(text)


if __name__ == "__main__":
    main()
