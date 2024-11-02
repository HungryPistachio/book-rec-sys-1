from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from xai.counterfactual_explanation import get_counterfactual_explanation
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as file:
        return file.read()

# LIME explanation endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    data = await request.json()
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books', [])
    try:
        explanation = get_lime_explanation(book_title, book_description, all_books)
        return JSONResponse(content=explanation)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# SHAP explanation endpoint
@app.post("/shap-explanation")
async def shap_explanation(request: Request):
    data = await request.json()
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books', [])
    try:
        explanation = get_shap_explanation(book_title, book_description, all_books)
        return JSONResponse(content=explanation)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Counterfactual explanation endpoint
@app.post("/counterfactual-explanation")
async def counterfactual_explanation(request: Request):
    data = await request.json()
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books', [])
    try:
        explanation = get_counterfactual_explanation(book_title, book_description, all_books)
        return JSONResponse(content=explanation)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
