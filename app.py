import uvicorn
from fastapi import FastAPI, Request, jsonify
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
from xai.counterfactual_explanation import get_counterfactual_explanation

app = FastAPI()

@app.post("/explain_xai")
async def explain_xai(request: Request):
    data = await request.json()
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
        return jsonify({"error": str(e)}), 500

    if explanation:
        return jsonify({"explanation": explanation['explanation']})
    else:
        return jsonify({"error": "Failed to generate explanation"}), 500

