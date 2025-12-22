from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import config.env  # Initialize environment variables
from utils.image_desc import describe_image
from workflow.graph import build_workflow
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)
CORS(app)
chatbot = build_workflow()

# Setup image upload path
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("ui.html")


@app.route("/chat", methods=["POST"])
def chat():
    text_input = request.form.get("message", "").strip()
    image_file = request.files.get("image")
    
    final_query = ""

    # If image is provided
    if image_file:
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        image_result = describe_image(filepath)

        if "description" in image_result:
            final_query = image_result["description"]
        else:
            return jsonify({"reply": f"Image processing error: {image_result['error']}"})
    
    # If text is provided (use either or both)
    if text_input:
        final_query = f"{text_input}. " + final_query if final_query else text_input

    if not final_query:
        return jsonify({"reply": "Please provide a question or an image."})

    input_data = {"question": HumanMessage(content=final_query)}
    result = chatbot.invoke(input=input_data, config={"configurable": {"thread_id": 3}})

    reply = result["messages"][-1].content if "messages" in result else "Sorry, something went wrong."
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
