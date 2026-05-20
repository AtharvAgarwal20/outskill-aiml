from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT = (
    "You are a helpful, friendly chatbot. Answer concisely unless the user "
    "asks for detail. Use Markdown when it improves clarity."
)


@app.route("/")
def index():
    return render_template("index.html", model=MODEL)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get("messages", [])

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    def generate():
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    yield f"data: {json.dumps({'token': piece})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/health")
def health():
    return jsonify({"ok": True, "model": MODEL})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1234, debug=True)
