from flask import Flask, request, jsonify, render_template
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
from xai.counterfactual_explanation import get_counterfactual_explanation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lime_explanation', methods=['POST'])
def lime_explanation():
    data = request.json
    book_title = data['book_title']
    book_description = data['book_description']
    all_titles = data['all_titles']
    all_descriptions = data['all_descriptions']
    
    explained_title, explanation = get_lime_explanation(book_title, book_description, all_titles, all_descriptions)
    
    return jsonify({
        "book_title": explained_title,
        "explanation": explanation
    })

@app.route('/shap_explanation', methods=['POST'])
def shap_explanation():
    data = request.json
    book_title = data['book_title']
    book_description = data['book_description']
    all_titles = data['all_titles']
    all_descriptions = data['all_descriptions']
    
    explanation = get_shap_explanation(book_title, book_description, all_titles, all_descriptions)
    
    return jsonify(explanation)

@app.route('/counterfactual_explanation', methods=['POST'])
def counterfactual_explanation():
    data = request.json
    book_title = data['book_title']
    book_description = data['book_description']
    all_titles = data['all_titles']
    all_descriptions = data['all_descriptions']
    
    counterfactuals = get_counterfactual_explanation(book_title, book_description, all_titles, all_descriptions)
    
    return jsonify(counterfactuals)

if __name__ == "__main__":
    app.run(debug=True)

