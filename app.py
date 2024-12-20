from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import json
from xai.lime_explanation import get_lime_explanation
from xai.anchor_explanations import get_anchor_explanation
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from pydantic import BaseModel

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model at the start
# try:
#     model = joblib.load('model/trained_model.joblib')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")


app = FastAPI()

# Mount static files for serving CSS and JS if they exist
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")


# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_feature_names = None  # Initialize as None

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

# Request body model
class DescriptionInput(BaseModel):
    description: str

@app.post("/generate-keywords")
async def generate_keywords(data: DescriptionInput):
    description = data.description.strip()

    if not description:
        raise HTTPException(status_code=400, detail="No description provided")

    # Generate keywords using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform([description])
    keywords = vectorizer.get_feature_names_out()

    return {"keywords": list(keywords)}



@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])
    logging.info(f"Received {len(descriptions)} descriptions for TF-IDF vectorization.")

    if not descriptions:
        logging.error("No descriptions provided.")
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Dynamically generate vocabulary based on descriptions
    vectorizer = TfidfVectorizer()  # No fixed vocabulary
    tfidf_matrix = vectorizer.fit_transform(descriptions).toarray()
    feature_names = vectorizer.get_feature_names_out()

    # Log details about the generated matrix
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logging.debug(f"Feature names: {feature_names}")

    # Respond with vectors for all descriptions
    return JSONResponse(content={
        "vectorized_descriptions": tfidf_matrix.tolist(),  # Full matrix of vectors
        "feature_names": feature_names.tolist(),
        "tfidf_matrix": tfidf_matrix.tolist()
    })



# LIME Explanation Endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    data = await request.json()
    recommendations = data.get("recommendations", [])  # List of recommendation details

    logging.info("Received request for LIME explanation.")

    try:
        explanation = get_lime_explanation(recommendations)
        logging.info("LIME explanations generated successfully.")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in LIME explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/anchor-explanation")
async def anchor_explanation(request: Request):
    data = await request.json()
    recommendations = data.get("recommendations", [])
    original_description = data.get("original_description", "")
    logging.info("Received request for Anchor explanation.")

    try:
        explanation = get_anchor_explanation(recommendations, original_description)
        logging.info(f"Response data: {explanation}")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in Anchor explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)




