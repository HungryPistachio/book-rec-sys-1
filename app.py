import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import JSONResponse
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
from xai.counterfactual_explanation import get_counterfactual_explanation

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as file:
        return file.read()

# @app.post("/explain_xai")
# async def explain_xai(request: Request):
#     data = await request.json()
#     model = data.get('model')
#     book_title = data.get('book_title')
#     book_description = data.get('book_description')
#     all_books = data.get('all_books')
#
#     explanation = None
#     try:
#         if model == 'lime':
#             explanation = get_lime_explanation(book_title, book_description, [book['description'] for book in all_books])
#         elif model == 'shap':
#             explanation = get_shap_explanation(book_title, book_description, [book['description'] for book in all_books])
#         elif model == 'counterfactual':
#             explanation = get_counterfactual_explanation(book_title, book_description, [book['description'] for book in all_books])
#         else:
#             return {"error": "Invalid model specified"}
#     except Exception as e:
#         return {"error": str(e)}
#     if explanation:
#         return {"explanation": explanation['explanation']}
#     else:
#         return {"error": "Failed to generate explanation"}
#
#      if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
