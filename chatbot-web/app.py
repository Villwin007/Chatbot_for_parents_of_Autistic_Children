from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from config.secret_config import API_KEY

# Configure the Generative AI
genai.configure(api_key=API_KEY)

app = Flask(__name__)

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

chat_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.json.get("message")
    session_id = request.json.get("session_id")

    # Initialize session if not already created
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "You are a helpful assistant for parents of autistic children. "
                        "Respond only to queries related to autism. For unrelated topics, reply: "
                        "'Please ask queries related to Autism only. I am not aware of any other information.' "
                        "Keep responses under 50 words.",
                    ],
                },
                {
                    "role": "model",
                    "parts": ["Okay. Ask me your questions about autism in children."],
                },
            ]
        )

    # Send user message to the chatbot
    chat_session = chat_sessions[session_id]
    try:
        response = chat_session.send_message(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
