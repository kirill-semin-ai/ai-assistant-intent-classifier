# AI Assistant Intent Classifier

A small NLP project for classifying user commands for an AI assistant.

The project is inspired by my JarvisCore AI-agent showcase. In JarvisCore, user requests can be routed to different modes: regular dialogue, source reading, technical commands, AI-agent tasks or safety checks. This project demonstrates a simple machine learning approach to intent classification.

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

---

## Planned Project Structure

```text
ai-assistant-intent-classifier/
  README.md
  data/
    intents.csv
  src/
    train.py
    predict.py
  requirements.txt
```

---

## Why This Project Matters

This project shows a basic NLP pipeline:

1. creating a small labeled dataset;
2. converting text into numeric features with TF-IDF;
3. training a classification model;
4. evaluating the model with metrics;
5. using the model to predict intents for new user messages.

It is a small but practical ML/NLP project connected to AI assistants, request routing and LLM-agent systems.

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
