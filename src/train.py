from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(**file**).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "intents.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"

def load_dataset(path: Path) -> pd.DataFrame:
"""Load and validate the intent classification dataset."""
if not path.exists():
raise FileNotFoundError(f"Dataset not found: {path}")

```
df = pd.read_csv(path)

required_columns = {"text", "intent"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.dropna(subset=["text", "intent"])
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

if df.empty:
    raise ValueError("Dataset is empty after cleaning.")

return df
```

def train_model(df: pd.DataFrame) -> Pipeline:
"""Train a simple TF-IDF + Logistic Regression classifier."""
x = df["text"]
y = df["intent"]

```
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("AI Assistant Intent Classifier")
print("=" * 40)
print(f"Dataset size: {len(df)} examples")
print(f"Train size: {len(x_train)} examples")
print(f"Test size: {len(x_test)} examples")
print()
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print()
print("Classification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
labels = sorted(df["intent"].unique())
print("Labels:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))

return model
```

def save_model(model: Pipeline, path: Path) -> None:
"""Save the trained model to disk."""
path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, path)
print()
print(f"Model saved to: {path}")

def main() -> None:
df = load_dataset(DATA_PATH)
model = train_model(df)
save_model(model, MODEL_PATH)

if **name** == "**main**":
main()
