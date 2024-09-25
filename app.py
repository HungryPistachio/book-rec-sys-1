from flask import Flask, render_template, request, jsonify
from xai.lime_explanation import get_lime_explanation  # Import explanation functions
from xai.shap_explanation import get_shap_explanation
from xai.counterfactual_explanation import get_counterfactual_explanation

app = Flask(__name__)

# Route to serve the index.html frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle XAI explanations
@app.route('/explain_xai', methods=['POST'])
def explain_xai():
    data = request.json
    model = data.get('model')
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books')

    # Debugging: Print the received data
    print(f"Received model: {model}, book_title: {book_title}, book_description: {book_description}")
    print(f"All books: {all_books}")

    explanation = None
    try:
        if model == 'lime':
            explanation = get_lime_explanation(book_title, book_description, [book['description'] for book in all_books])
        elif model == 'shap':
            explanation = get_shap_explanation(book_title, book_description, [book['description'] for book in all_books])
        elif model == 'counterfactual':
            explanation = get_counterfactual_explanation(book_title, book_description, [book['description'] for book in all_books])
        else:
            return jsonify({"error": "Invalid model specified"}), 400
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

    if explanation:
        return jsonify({"explanation": explanation['explanation']})
    else:
        return jsonify({"error": "Failed to generate explanation"}), 500
