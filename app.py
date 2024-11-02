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
    filtered_tfidf_matrix = data.get('filteredTfidfMatrix')  # Update to match the sent name
    feature_names = data.get('feature_names')

    # Logging for debugging
    logging.info(f"Book Title: {book_title}")
    logging.info(f"Description Vector (length): {len(description_vector) if description_vector else 'None'}")
    logging.info(f"Filtered TF-IDF Matrix (shape): {len(filtered_tfidf_matrix) if filtered_tfidf_matrix else 'None'} by {len(filtered_tfidf_matrix[0]) if filtered_tfidf_matrix else 'None'}")
    logging.info(f"Feature Names (count): {len(feature_names) if feature_names else 'None'}")

    if not all([book_title, description_vector, filtered_tfidf_matrix, feature_names]):
        return JSONResponse(content={"error": "Missing required fields in the request data"}, status_code=400)

    try:
        explanation = get_lime_explanation(book_title, description_vector, filtered_tfidf_matrix, feature_names)
        return JSONResponse(content=explanation)
    except Exception as e:
        logging.error(f"Error in /lime-explanation endpoint: {e}")
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
