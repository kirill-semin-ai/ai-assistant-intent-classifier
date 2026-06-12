# AI Assistant Intent Classifier

A small NLP project for classifying user commands for an AI assistant.

The project is inspired by my JarvisCore AI-agent showcase. In JarvisCore, user requests can be routed to different modes: regular dialogue, information search, source reading, technical commands, AI-agent tasks or safety checks. This project demonstrates a simple machine learning approach to intent classification for AI-assistant command routing.

---

## Project Goal

The goal of this project is to build a simple text classifier that predicts the intent of a user message.

Example:

```text
"find information about neural networks" -> information_search
"read this article and summarize it" -> source_reading
"delete all files" -> unsafe_request
"start an AI-agent task" -> ai_agent_task
```

The project is not intended to be a production-ready classifier. It is a small portfolio project that demonstrates the main stages of an NLP workflow: dataset creation, text vectorization, model training, evaluation, model saving and inference.

---

## Intent Classes

The classifier predicts one of the following categories:

* `regular_dialogue` — normal conversation with the assistant;
* `information_search` — request to search for information;
* `source_reading` — request to read or summarize a source;
* `technical_command` — request related to system status, files or technical actions;
* `ai_agent_task` — request to start or continue an AI-agent workflow;
* `unsafe_request` — potentially unsafe or sensitive request.

---

## Tech Stack

* Python
* pandas
* scikit-learn
* TF-IDF
* Logistic Regression
* train/test split
* classification metrics
* joblib

---

## Project Structure

```text
ai-assistant-intent-classifier/
  README.md
  requirements.txt
  data/
    intents.csv
  src/
    train.py
    predict.py
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/kirill-semin-ai/ai-assistant-intent-classifier.git
cd ai-assistant-intent-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

The training script will:

* load `data/intents.csv`;
* clean and validate the dataset;
* split the dataset into train and test parts;
* transform text into TF-IDF features;
* train a Logistic Regression classifier;
* print accuracy, classification report and confusion matrix;
* save the trained model to `models/intent_classifier.joblib`.

### 5. Predict intent for a new message

```bash
python src/predict.py "find information about neural networks"
```

Example output:

```text
AI Assistant Intent Classifier
========================================
Input text: find information about neural networks
Predicted intent: information_search
```

---

## Model

The project uses a simple baseline ML pipeline:

```text
Text -> TF-IDF -> Logistic Regression -> Intent class
```

TF-IDF is used to convert text messages into numeric features. Logistic Regression is then trained to classify each message into one of the intent classes.

This approach is simple, fast and suitable as a first baseline for a small text classification project.

---

## Current Results

After expanding the dataset to 120 labeled examples, the baseline model achieved the following result on the test split:

```text
Accuracy: 0.833
Macro avg F1-score: 0.83
Weighted avg F1-score: 0.83
```

Per-class results:

```text
                    precision    recall  f1-score   support

     ai_agent_task       0.75      1.00      0.86         6
information_search       1.00      1.00      1.00         6
  regular_dialogue       0.80      0.67      0.73         6
    source_reading       0.71      0.83      0.77         6
 technical_command       0.80      0.67      0.73         6
    unsafe_request       1.00      0.83      0.91         6

          accuracy                           0.83        36
         macro avg       0.84      0.83      0.83        36
      weighted avg       0.84      0.83      0.83        36
```

Example predictions:

```text
"find information about neural networks" -> information_search
"read this article and summarize it" -> source_reading
"delete all files permanently" -> unsafe_request
"check system status" -> technical_command
```

---

## Why This Project Matters

This project shows a complete small NLP pipeline:

1. creating a labeled dataset;
2. preparing user messages for classification;
3. converting text into numeric TF-IDF features;
4. training a baseline classification model;
5. evaluating the model with classification metrics;
6. saving the trained model;
7. using the model for inference on new user messages.

The task is connected to AI assistants, request routing and LLM-agent systems. A similar intent classifier can be used as one of the routing components in an AI assistant: before a request is sent to a specific tool, agent workflow or safety layer, the system can classify what type of request it is.

---

## Notes

The trained model is saved locally after running:

```bash
python src/train.py
```

The model file is saved to:

```text
models/intent_classifier.joblib
```

The `models/` directory is ignored by Git because trained model files are generated artifacts.

Model files such as `.joblib`, `.pkl` or `.pickle` should only be loaded when they come from a trusted source.

---

## Limitations

This is a baseline educational project with a small dataset.

Current limitations:

* the dataset is manually created and relatively small;
* the model may confuse similar classes such as `regular_dialogue` and `source_reading`;
* the classifier is trained only on short English examples;
* the project does not yet include cross-validation;
* the project does not yet include model comparison.

---

## Future Improvements

Planned improvements:

* add more training examples;
* add Russian-language examples;
* add confusion matrix visualization;
* add error analysis;
* compare Logistic Regression with Naive Bayes and Linear SVM;
* add cross-validation;
* add a simple command-line prediction interface;
* add a small FastAPI endpoint;
* later create a PyTorch or Transformers-based version.

---

## Related Project

This project is connected to my JarvisCore showcase:

* local AI-agent platform;
* request routing;
* local LLM integration;
* source reader;
* safe execution;
* smoke/regression testing;
* snapshot/rollback workflow.

JarvisCore focuses on AI-agent architecture and safe local assistant workflows. This repository focuses on a small ML/NLP component that can support AI-assistant request routing.
