from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & preprocessors
model = load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load FAQ responses
faq_df = pd.read_csv("faq_dataset.csv")

max_len = 20

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower()

    # Preprocess message
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # Predict intent
    pred = model.predict(padded)
    intent_idx = np.argmax(pred)
    predicted_intent = label_encoder.inverse_transform([intent_idx])[0]

    # Get response
    bot_reply = faq_df[faq_df['intent'] == predicted_intent]['bot_response'].iloc[0]

    return jsonify({"intent": predicted_intent, "response": bot_reply})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
