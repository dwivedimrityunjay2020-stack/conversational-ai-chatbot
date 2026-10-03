import json
import logging
import random
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "intents.json"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "chatbot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def download_nltk_data():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]
    return " ".join(tokens)

def load_intents():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)["intents"]

class NLPChatbot:
    def __init__(self):
        self.intents = load_intents()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.confidence_threshold = 0.38
        self._train()

    def _train(self):
        texts, labels = [], []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                texts.append(normalize(pattern))
                labels.append(intent["tag"])

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        logging.info("Chatbot model trained with %d examples.", len(texts))

    def predict(self, message):
        cleaned = normalize(message)
        if not cleaned.strip():
            return "fallback", 0.0

        X = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(X)[0]
        best_index = probabilities.argmax()
        intent = self.model.classes_[best_index]
        confidence = float(probabilities[best_index])

        if confidence < self.confidence_threshold:
            return "fallback", confidence
        return intent, confidence

    def respond(self, message):
        tag, confidence = self.predict(message)

        if tag == "fallback":
            response = (
                "I'm not fully sure what you mean. "
                "Try asking about greetings, hours, courses, fees, contact details, "
                "or type 'help' to see examples."
            )
            logging.info("FALLBACK | confidence=%.3f | user=%r", confidence, message)
            return response, tag, confidence

        intent = next(item for item in self.intents if item["tag"] == tag)
        response = random.choice(intent["responses"])
        logging.info(
            "INTENT=%s | confidence=%.3f | user=%r",
            tag, confidence, message
        )
        return response, tag, confidence

def main():
    bot = NLPChatbot()
    print("=" * 58)
    print("        NLP Conversational AI Chatbot")
    print("=" * 58)
    print("Type 'help' for examples or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! Have a great day.")
            break

        if user_input.lower() == "help":
            print("Bot: Try: 'hello', 'what courses do you offer?',")
            print("     'what are your fees?', 'when are you open?',")
            print("     'how can I contact you?', or ask something unrelated.")
            continue

        response, tag, confidence = bot.respond(user_input)
        print(f"Bot: {response}")
        print(f"     [intent: {tag}, confidence: {confidence:.2f}]\n")

if __name__ == "__main__":
    main()
