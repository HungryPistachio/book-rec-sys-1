from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Mount static files for serving CSS and JS if they exist
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path("templates/index.html")  # Adjust path if necessary
    if not index_path.exists():
        logging.error("index.html not found!")
        return JSONResponse(content={"error": "index.html not found"}, status_code=404)

    with open(index_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# TF-IDF Vectorization Endpoint
@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])
    logging.info("Received descriptions for TF-IDF vectorization.")

    if not descriptions:
        logging.error("No descriptions provided.")
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Vectorize descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions).toarray()
    feature_names = vectorizer.get_feature_names_out()
    description_vector = tfidf_matrix[0].tolist()  # Assume first description is target

    logging.info("TF-IDF vectorization complete.")
    return JSONResponse(content={
        "tfidf_matrix": tfidf_matrix.tolist(),
        "feature_names": feature_names.tolist(),
        "description_vector": description_vector
    })

# LIME Explanation Endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    data = await request.json()
    book_title = data.get("book_title")
    book_description = data.get("book_description")
    #all_books = data.get("all_books", [])

    logging.info(f"Starting LIME explanation for '{book_title}'")

    try:
        explanation = get_lime_explanation(book_title, book_description)
        logging.info("LIME explanation generated successfully.")
        return JSONResponse(content=explanation)
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# SHAP Explanation Endpoint
@app.post("/shap-explanation")
async def shap_explanation(request: Request):
    data = await request.json()
    book_title = data.get("book_title")
    book_description = data.get("book_description")
    #all_books = data.get("all_books", [])

    logging.info(f"Starting SHAP explanation for '{book_title}'")

    try:
        explanation = get_shap_explanation(book_title, book_description)
        logging.info("SHAP explanation generated successfully.")
        return JSONResponse(content=explanation)
    except Exception as e:
        logging.error(f"Error in SHAP explanation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
