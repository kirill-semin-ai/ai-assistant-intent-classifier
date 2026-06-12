# AI Assistant Intent Classifier

A small NLP project for classifying user commands for an AI assistant.

The project is inspired by my JarvisCore AI-agent showcase. In JarvisCore, user requests can be routed to different modes: regular dialogue, information search, source reading, technical commands, AI-agent tasks or safety checks. This project demonstrates a simple machine learning approach to intent classification.

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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

The training script will:

* load `data/intents.csv`;
* split the dataset into train and test parts;
* transform text into TF-IDF features;
* train a Logistic Regression classifier;
* print accuracy, classification report and confusion matrix;
* save the trained model to `models/intent_classifier.joblib`.

### 4. Predict intent for a new message

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

This is a baseline model, not a production-ready classifier. The goal is to demonstrate a complete small NLP workflow: dataset creation, text vectorization, model training, evaluation and prediction.

---

## Why This Project Matters

This project shows a basic NLP pipeline:

1. creating a small labeled dataset;
2. converting text into numeric features with TF-IDF;
3. training a classification model;
4. evaluating the model with classification metrics;
5. using the model to predict intents for new user messages.

It is a small but practical ML/NLP project connected to AI assistants, request routing and LLM-agent systems.

---

## Notes

The trained model is saved locally after running:

```bash
python src/train.py
```

Model files such as `.joblib` should only be loaded when they come from a trusted source. Serialized model files can be unsafe if they are downloaded from unknown or untrusted repositories.

---

## Future Improvements

Planned improvements:

* add more training examples;
* add confusion matrix and error analysis;
* compare Logistic Regression with Naive Bayes or Linear SVM;
* add a simple command-line prediction script;
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
