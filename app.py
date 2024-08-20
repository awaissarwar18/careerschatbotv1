from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = Flask(__name__)

# Load the model and tokenizer locally
model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

# Create a text-generation pipeline using the locally loaded model and tokenizer
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Function to get AI response
def get_ai_response(query):
    try:
        # Generate a response using GPT-2
        response = generator(query, max_length=100, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Route to handle the root URL and serve the HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Route to handle the chatbot queries
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("query")
    if user_input:
        response = get_ai_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Please ask a question about a career."})

if __name__ == '__main__':
    app.run(debug=True)
