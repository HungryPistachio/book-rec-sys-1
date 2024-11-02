from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
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

    return JSONResponse(content={
        "tfidf_matrix": tfidf_matrix.tolist(),
        "feature_names": feature_names.tolist(),
        "description_vector": description_vector
    })
# LIME explanation endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    try:
        # Log request data
        data = await request.json()
        print("Data received:", data)

        # Extract relevant fields
        data = await request.json()
        book_title = data.get('book_title')
        book_description = data.get('book_description')
        all_books = data.get('all_books', [])
        tfidf_matrix = data.get('tfidf_matrix')
        feature_names = data.get('feature_names')

        # Log extracted fields
        print("Book Title:", book_title)
        print("Book Description:", book_description)
        print("All Books:", all_books)

        # Ensure all required data is present
        if not book_title or not book_description or not all_books:
            raise ValueError("Missing required fields in the request data")

        # Call the LIME explanation function and return its result
        explanation = get_lime_explanation(book_title, book_description, all_books, tfidf_matrix, feature_names)
        return JSONResponse(content=explanation)


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
