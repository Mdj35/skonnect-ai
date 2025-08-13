from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from pyngrok import ngrok
from threading import Thread

NGROK_AUTH_TOKEN = "2myGpjp7qIUGyA03PrkdSAw5nCy_4dNyxuyPHzLNgxY7Bk8f1"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

with open("chatbot_model.pkl", "rb") as f:
    chatbot_model = pickle.load(f)

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

public_url = ngrok.connect(5000)
print(f" * ngrok tunnel available at: {public_url}")

def run():
    app.run(port=5000)

Thread(target=run).start()
