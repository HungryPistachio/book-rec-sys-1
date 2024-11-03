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
        # Convert feature names to indices to simulate matching description_vector
        predictions = []
        for text in text_input:
            # This converts words into "importance scores" based on their indices in description_vector
            vectorized_input = [description_vector[feature_names.index(word)] if word in feature_names else 0 for word in text.split()]
            prediction = np.array([np.sum(vectorized_input)])
            predictions.append(prediction)
        prediction_array = np.array(predictions)
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
            num_features=len(feature_names)
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

