from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

# Load model
with open("chatbot_model.pkl", "rb") as f:
    chatbot_model = pickle.load(f)

# Load FAQ dataset
faq_df = pd.read_csv("faq_dataset.csv")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    predicted_intent = chatbot_model.predict([message])[0]
    bot_reply = faq_df[faq_df['intent'] == predicted_intent]['bot_response'].iloc[0]
    return jsonify({"intent": predicted_intent, "response": bot_reply})

if __name__ == '__main__':
    # Railway sets the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
