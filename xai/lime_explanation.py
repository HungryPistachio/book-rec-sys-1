import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stop words

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(description_vector, feature_names):
    logging.info("Starting LIME explanation generation.")

    # Ensure description_vector and feature_names are properly structured
    if not isinstance(description_vector, list) or not isinstance(feature_names, list):
        logging.error("Invalid input format for LIME explanation.")
        return json.dumps({"error": "Invalid input format"})

    try:
        explainer = LimeTextExplainer(class_names=['Book'])

        # Create input text and define explanation function
        input_text = ' '.join(feature_names)  # Mock input text for LIME
        explanation = explainer.explain_instance(
            input_text,
            lambda x: np.array([description_vector] * len(x)),  # Broadcast description vector
            num_features=len(feature_names)
        )
        stop_words = ["this", "for", "have", "and", "a", "is", "are", "______________________________", "be", "without", "made", "when", "thing", "to", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        explanation_output = [
            (word, weight) for word, weight in explanation.as_list()
            if word.lower() not in stop_words  # Filter out stop words
        ]
        logging.info("LIME explanation generated successfully.")
        print("LIME explanations identify the feature contributions to the recommendation.")

        response = {
            "general_explanation": "LIME explanation for the book recommendation.",
            "explanation_output": explanation_output
        }
        return json.dumps(response)

    except Exception as e:
        logging.error(f"Error generating LIME explanation: {e}")
        return json.dumps({"error": "LIME explanation generation failed"})
