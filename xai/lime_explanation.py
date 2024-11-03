import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(description_vector, feature_names):
    logging.info("Starting LIME explanation generation.")

    # Initialize the LimeTextExplainer for textual data
    explainer = LimeTextExplainer(class_names=['Book'])

    # Define a classifier function that mimics model output based on description_vector
    def predict_fn(text_input):
        predictions = []
        for text in text_input:
            # Convert text to feature importance scores based on description_vector
            vectorized_input = [
                description_vector[feature_names.index(word)] if word in feature_names else 0
                for word in text.split()
            ]
            # Sum the vector to simulate a single prediction value for each input
            prediction = np.array([sum(vectorized_input)])
            predictions.append(prediction)

        prediction_array = np.array(predictions)
        # Reshape to ensure it matches expected dimensions (num_samples, 1)
        if prediction_array.ndim == 1:
            prediction_array = prediction_array.reshape(-1, 1)
        logging.debug(f"Generated predictions: {prediction_array}")
        return prediction_array

    # Combine feature names into a single representative text input for LIME
    text_input = ' '.join(feature_names)
    logging.debug(f"Generated text input for LIME: {text_input}")

    try:
        # Generate explanation using the LimeTextExplainer
        explanation = explainer.explain_instance(
            text_instance=text_input,
            classifier_fn=predict_fn,
            num_features=min(len(feature_names), 10)  # Limit to 10 features for faster processing
        )
        
        explanation_output = explanation.as_list()
        logging.info("LIME explanation generated successfully.")

        # Construct the response
        response = {
            "general_explanation": "LIME explanation for the book recommendation.",
            "explanation_output": explanation_output
        }
        return json.dumps(response)
    except Exception as e:
        logging.error(f"Error generating LIME explanation: {e}")
        return json.dumps({"error": "Failed to generate LIME explanation"})

