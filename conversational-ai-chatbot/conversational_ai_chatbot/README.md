# Conversational AI Chatbot — Python + NLP

An interactive NLP-based chatbot that uses **TF-IDF**, **n-gram features**, and **Logistic Regression** for intent classification. It includes confidence-based fallback handling and interaction logging.

## Features

- NLP text normalization
- Stop-word removal and lemmatization
- TF-IDF vectorization
- Unigram + bigram features
- Intent classification with Logistic Regression
- Confidence threshold for ambiguous queries
- Fallback response for out-of-scope questions
- Timestamped interaction logging
- JSON-based intent dataset
- Simple command-line interface

## Project Structure

```text
conversational-ai-chatbot/
├── chatbot.py
├── data/
│   └── intents.json
├── logs/
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- pip

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/conversational-ai-chatbot.git
cd conversational-ai-chatbot
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the chatbot

```bash
python chatbot.py
```

The first run downloads the required NLTK resources automatically.

## Example

```text
You: hello
Bot: Hello! How can I help you?

You: what courses do you offer?
Bot: You can ask about Computer Science, AI/ML, or Data Analytics programs.

You: tell me today's weather
Bot: I'm not fully sure what you mean...
```

## How It Works

```text
User Input
    ↓
Text Cleaning
    ↓
Tokenization / Stop-word Removal
    ↓
Lemmatization
    ↓
TF-IDF + N-gram Features
    ↓
Logistic Regression
    ↓
Intent + Confidence
    ↓
 ┌───────────────┐
 │ Confidence OK │──→ Intent Response
 └───────────────┘
          │
          ↓
    Low Confidence
          │
          ↓
       Fallback
          │
          ↓
      Interaction Log
```

## Customization

Add new intents inside `data/intents.json`. Each intent contains:

- `tag`: intent name
- `patterns`: example user messages
- `responses`: possible chatbot responses

For a better classifier, add several varied patterns for every intent.

## Resume Description

**Conversational AI Chatbot | Python, NLP, Scikit-learn**
- Engineered an interactive NLP chatbot using TF-IDF features and Logistic Regression for intent-based classification.
- Implemented text normalization, stop-word removal, lemmatization, n-gram features, and confidence-based fallback handling.
- Added timestamped interaction logging to track classified and ambiguous user queries.

## Future Enhancements

- Flask/FastAPI web interface
- SQLite conversation database
- Speech-to-text and text-to-speech
- Transformer/BERT-based intent classification
- Admin dashboard for conversation analytics
- Deployment on a cloud platform
