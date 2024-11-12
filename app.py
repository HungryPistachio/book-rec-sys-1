from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import json
from xai.lime_explanation import get_lime_explanation
from xai.dice_explanation import get_dice_explanation
from xai.dice_explanation import initialize_dice
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the trained model at the start
try:
    model = joblib.load('model/trained_model.joblib')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the fixed vocabulary once on startup
vocabulary_path = Path("static/fixed_vocabulary.csv")
if vocabulary_path.exists():
    feature_names = pd.read_csv(vocabulary_path)["Vocabulary"].tolist()
    print("Fixed vocabulary loaded successfully.")
else:
    feature_names = None
    print("Error: fixed_vocabulary.csv not found.")


app = FastAPI()

# Mount static files for serving CSS and JS if they exist
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")


# Initialize the TF-IDF Vectorizer


def load_fixed_vocabulary():
    vocab_df = pd.read_csv("static/fixed_vocabulary.csv")
    return vocab_df["Vocabulary"].tolist()

fixed_vocabulary = load_fixed_vocabulary()
dice = initialize_dice(model, feature_names)
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


# Global variable to store TF-IDF feature names
@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])
    logging.info("Received descriptions for TF-IDF vectorization.")

    if not descriptions:
        logging.error("No descriptions provided.")
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Use TfidfVectorizer with the fixed vocabulary
    vectorizer = TfidfVectorizer(vocabulary=fixed_vocabulary)
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
    recommendations = data.get("recommendations", [])  # List of recommendation details

    logging.info("Received request for LIME explanation.")

    try:
        explanation = get_lime_explanation(recommendations)
        logging.info("LIME explanations generated successfully.")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in LIME explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Dice Explanation Endpoint
@app.post("/dice-explanation")
async def dice_explanation(request: Request):
    data = await request.json()
    recommendations = data.get("recommendations", [])
    logging.info("Received request for Dice explanation.")

    if feature_names is None:
        logging.error("TF-IDF feature names not available; ensure fixed_vocabulary.csv is loaded.")
        return JSONResponse(content={"error": "TF-IDF feature names not available; run vectorize-descriptions first."}, status_code=400)

    try:
        description_vector = recommendations[0]["description_vector"]

        input_data = pd.DataFrame([description_vector], columns=feature_names)
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        explanation = get_dice_explanation(dice, input_data, feature_names)
        logging.info("Dice explanations generated successfully.")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in Dice explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


