from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========================
# Load model & preprocessors
# ========================
model = load_model("chatbot_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load FAQ dataset
faq_df = pd.read_csv("faq_dataset.csv")

max_len = 20
LOG_FILE = "chat_logs.csv"

# ========================
# Ensure log file exists
# ========================
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "user_message", "predicted_intent", "bot_response"]).to_csv(LOG_FILE, index=False)

def log_conversation(user_message, predicted_intent, bot_reply):
    """Append a single conversation entry to the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([[timestamp, user_message, predicted_intent, bot_reply]],
                          columns=["timestamp", "user_message", "predicted_intent", "bot_response"])
    log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# ========================
# Flask App Setup
# ========================
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint: takes user message and returns predicted intent + response."""
    data = request.json
    message = data.get('message', '').lower()

    # Preprocess message
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # Predict intent
    pred = model.predict(padded)
    intent_idx = np.argmax(pred)
    predicted_intent = label_encoder.inverse_transform([intent_idx])[0]

    # Get response (default if not found)
    bot_reply = "I'm not sure how to respond to that yet."
    if predicted_intent in faq_df['intent'].values:
        bot_reply = faq_df[faq_df['intent'] == predicted_intent]['bot_response'].iloc[0]

    # Log conversation
    log_conversation(message, predicted_intent, bot_reply)

    return jsonify({"intent": predicted_intent, "response": bot_reply})

# ========================
# Run the App
# ========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
