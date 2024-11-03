from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from xai.lime_explanation import get_lime_explanation
from xai.shap_explanation import get_shap_explanation

logging.basicConfig(level=logging.INFO)

app = FastAPI()

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
