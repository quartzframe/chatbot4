import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

MODEL_NAME = "gemini-3.1-flash-lite"


def load_chatbot_config():
    config_path = os.path.join(os.path.dirname(__file__), "chatbot_config.txt")
    with open(config_path, "r", encoding="utf-8") as file:
        return file.read().strip()


SYSTEM_PROMPT = load_chatbot_config()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    if not client:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    # Keep only valid, recent text turns to control request size.
    contents = []
    if isinstance(history, list):
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            text = str(item.get("text", "")).strip()
            if role in {"user", "model"} and text:
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=text)])
                )

    contents.append(
        types.Content(role="user", parts=[types.Part(text=message)])
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.35,
                max_output_tokens=1200,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't generate a response right now. Please try again."

        return jsonify({"answer": answer, "model": MODEL_NAME})
    except Exception as exc:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "The AI service could not process your request."}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
