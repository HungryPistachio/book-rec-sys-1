from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import json
from xai.lime_explanation import get_lime_explanation
from xai.dice_explanation import get_dice_explanation
from xai.dice_explanation import initialize_dice
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import pad_missing_columns  # Import pad_missing_columns from utils
import uvicorn
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model at the start
try:
    model = joblib.load('model/trained_model.joblib')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the fixed vocabulary once on startup
# Load the fixed vocabulary from CSV
vocab_path = "static/fixed_vocabulary.csv"
try:
    fixed_vocabulary = pd.read_csv(vocab_path)["Vocabulary"].tolist()
    logging.info("Fixed vocabulary loaded successfully.")
except Exception as e:
    logging.error(f"Error loading fixed vocabulary: {e}")
    fixed_vocabulary = None



app = FastAPI()

# Mount static files for serving CSS and JS if they exist
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")


# Initialize the TF-IDF Vectorizer


def load_fixed_vocabulary():
    vocab_df = pd.read_csv("static/fixed_vocabulary.csv")
    return vocab_df["Vocabulary"].tolist()

fixed_vocabulary = load_fixed_vocabulary()
dice = initialize_dice(model, fixed_vocabulary)
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
# Original vectorize_descriptions endpoint (without fixed vocabulary)
# Updated vectorize_descriptions function output to maintain compatibility
@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])
    logging.info("Received descriptions for TF-IDF vectorization.")

    if not descriptions:
        logging.error("No descriptions provided.")
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Dynamically generate vocabulary based on descriptions
    vectorizer = TfidfVectorizer()  # No fixed vocabulary
    tfidf_matrix = vectorizer.fit_transform(descriptions).toarray()
    feature_names = vectorizer.get_feature_names_out()
    vectorized_descriptions = tfidf_matrix[0].tolist()

    logging.info("TF-IDF vectorization complete.")
    return JSONResponse(content={
        "vectorized_descriptions": vectorized_descriptions,
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


@app.post("/dice-explanation")
async def dice_explanation(request: Request):
    data = await request.json()
    recommendations = data.get("recommendations", [])
    logging.info("Received request for Dice explanation.")

    try:
        # Get the vectorized description
        vectorized_description = recommendations[0]["vectorized_description"]

        # Create a DataFrame with the correct number of columns
        input_data = pd.DataFrame([vectorized_description], columns=fixed_vocabulary)

        # Ensure all feature values are numeric
        input_data = input_data.apply(pd.to_numeric)

        # Generate counterfactual explanation
        explanation = get_dice_explanation(dice, input_data)
        logging.info("Dice explanations generated successfully.")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in Dice explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)





