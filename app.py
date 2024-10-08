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
    app.logger.info("Received POST request for explanation")  # Log the request arrival
    data = request.json
    
    app.logger.info(f"Data received: {data}")  # Log the data received

    model = data.get('model')
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books')

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
        app.logger.error(f"Error generating explanation: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

    if explanation:
        return jsonify({"explanation": explanation['explanation']})
    else:
        return jsonify({"error": "Failed to generate explanation"}), 500

if __name__ == '__main__':
    app.run(debug=True)

