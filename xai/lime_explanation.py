import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stop words

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(recommendations):
    logging.info("Starting LIME explanation generation for multiple recommendations.")
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['Book'])
    explanations = []
    # Loop through each recommendation
    for idx, rec in enumerate(recommendations):
        try:
            description_vector = rec.get("description_vector", [])
            feature_names = rec.get("feature_names", [])
            input_text = ' '.join(feature_names[:500])  # Limit input text for LIME
            # Generate explanation using LIME
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),
                num_features=min(len(feature_names), 100)
            )
            # Filter for significant features only
            stop_words = ["this", "for", "have", "and", "a", "is", "are", "______________________________", "be", "without", "made", "when", "thing", "to", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

            explanation_output = [
                (word, weight) for word, weight in explanation.as_list()
                if word.lower() not in stop_words  # Filter out stop words
]
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "general_explanation": "LIME explanation for the book recommendation.",
                "explanation_output": explanation_output
            })
        except Exception as e:
            logging.error(f"Error generating LIME explanation for recommendation {idx + 1}: {e}")
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "general_explanation": "Failed to generate explanation.",
                "explanation_output": []
            })
    logging.info("LIME explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
