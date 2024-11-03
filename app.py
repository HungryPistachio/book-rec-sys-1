from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import logging
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Vectorize descriptions endpoint
@app.post("/vectorize-descriptions")
async def vectorize_descriptions(request: Request):
    data = await request.json()
    descriptions = data.get("descriptions", [])

    if not descriptions:
        logging.error("No descriptions provided.")
        return JSONResponse(content={"error": "No descriptions provided"}, status_code=400)

    # Vectorize descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions).toarray()
    feature_names = vectorizer.get_feature_names_out()
    description_vector = tfidf_matrix[0].tolist()  # Assuming the first description is the target

    logging.info("TF-IDF vectorization completed.")
    logging.debug(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    logging.debug(f"Feature Names: {feature_names}")

    return JSONResponse(content={
        "tfidf_matrix": tfidf_matrix.tolist(),
        "feature_names": feature_names.tolist(),
        "description_vector": description_vector
    })

# LIME explanation endpoint
@app.post("/lime-explanation")
async def lime_explanation(request: Request):
    data = await request.json()
    description_vector = data.get('description_vector')
    feature_names = data.get('feature_names')

    if not description_vector or not feature_names:
        logging.error("Missing required fields for LIME explanation.")
        return JSONResponse({"error": "Missing required fields"}, status_code=400)

    explanation = get_lime_explanation(description_vector, feature_names)
    return JSONResponse(content=json.loads(explanation))

# SHAP explanation endpoint
@app.post("/shap-explanation")
async def shap_explanation(request: Request):
    data = await request.json()
    description_vector = data.get('description_vector')
    feature_names = data.get('feature_names')

    if not description_vector or not feature_names:
        logging.error("Missing required fields for SHAP explanation.")
        return JSONResponse({"error": "Missing required fields"}, status_code=400)

    explanation = get_shap_explanation(description_vector, feature_names)
    return JSONResponse(content=json.loads(explanation))
