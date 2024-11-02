from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
from xai.counterfactual_explanation import get_counterfactual_explanation
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation
import logging
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as file:
        return file.read()

@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])

    if not descriptions:
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Vectorize all descriptions at once
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions).toarray()
    feature_names = vectorizer.get_feature_names_out()
    description_vector = tfidf_matrix[0].tolist()  # Assuming the first description is the target

    # Log matrix shape for debugging
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

    return JSONResponse(content={
        "tfidf_matrix": tfidf_matrix.tolist(),
        "feature_names": feature_names.tolist(),
        "description_vector": description_vector
    })
# LIME explanation endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    data = await request.json()
    book_title = data.get('book_title')
    description_vector = data.get('description_vector')
    tfidf_matrix = data.get('tfidf_matrix')
    feature_names = data.get('feature_names')

    # Logging data received for debugging
    logging.info("Data received for LIME explanation:")
    logging.info(f"Book Title: {book_title}")
    logging.info(f"Description Vector: {description_vector}")
    logging.info(f"TF-IDF Matrix: {tfidf_matrix}")
    logging.info(f"Feature Names: {feature_names}")

    # Check if any of the required fields are None and raise an error
    if book_title is None or description_vector is None or tfidf_matrix is None or feature_names is None:
        logging.error("Missing required fields in the request data for LIME explanation")
        raise HTTPException(status_code=400, detail="Missing required fields in the request data.")

    # Convert tfidf_matrix to an appropriate format if necessary
    if isinstance(tfidf_matrix, list):
        import numpy as np
        tfidf_matrix = np.array(tfidf_matrix)

    # Ensure description_vector is the right format
    if isinstance(description_vector, list):
        description_vector = np.array(description_vector)

    try:
        explanation = get_lime_explanation(book_title, description_vector, tfidf_matrix, feature_names)
        return JSONResponse(content=explanation)
    except Exception as e:
        logging.error(f"Error in /lime-explanation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


    except Exception as e:
        # Log the error and respond with a 500 status
        print("Error in /lime-explanation endpoint:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

# SHAP explanation endpoint
@app.post("/shap-explanation")
async def shap_explanation(request: Request):
    data = await request.json()
    book_title = data.get('book_title')
    book_description = data.get('book_description')
    all_books = data.get('all_books', [])
    tfidf_matrix = data.get('tfidf_matrix')
    feature_names = data.get('feature_names')
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
    tfidf_matrix = data.get('tfidf_matrix')
    feature_names = data.get('feature_names')
    try:
        explanation = get_counterfactual_explanation(book_title, book_description, all_books)
        return JSONResponse(content=explanation)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
