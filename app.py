from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

# Load model & vectorizer
with open("chatbot_model.pkl", "rb") as f:
    chatbot_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load FAQ dataset
faq_df = pd.read_csv("faq_dataset.csv")

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower()

    # Transform message to vector
    X_input = vectorizer.transform([message])

    # Predict intent
    predicted_intent = chatbot_model.predict(X_input)[0]

    # Find response
    bot_reply = faq_df[faq_df['intent'] == predicted_intent]['bot_response'].iloc[0]

    return jsonify({"intent": predicted_intent, "response": bot_reply})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
