import json
import spacy
import logging
from alibi.explainers import AnchorText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the AnchorText explainer from Alibi with placeholder prediction function
explainer = AnchorText(nlp, predictor=None, use_unk=True)

# Function to set up the predictor function based on the initial book description
def setup_predictor(original_description, descriptions):
    # Vectorize the original description along with recommendations
    vectorizer = TfidfVectorizer()
    all_descriptions = [original_description] + descriptions
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    # Calculate cosine similarity of each recommendation to the original
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Return binary predictions based on similarity threshold
    return [1 if score >= 0.5 else 0 for score in similarity_scores]

def get_anchor_explanation(recommendations, original_description):
    explanations = []
    descriptions = [rec.get("description", "") for rec in recommendations]
    
    # Define the predictor function dynamically based on similarity with original description
    def predict_fn(texts):
        return setup_predictor(original_description, texts)
    
    # Attach the predictor to the explainer
    explainer.predictor = predict_fn

    for idx, rec in enumerate(recommendations):
        try:
            description = rec.get("description", "")
            if not description:
                logging.warning(f"No description found for recommendation {idx + 1}")
                continue

            # Generate an explanation
            explanation = explainer.explain(description, threshold=0.95)

            # Collect anchor words and precision score
            anchor_words = " AND ".join(explanation.data['anchor'])
            precision = explanation.data['precision']

            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "anchor_words": anchor_words,
                "precision": precision
            })
        except Exception as e:
            logging.error(f"Error generating Anchor explanation for recommendation {idx + 1}: {e}")
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "anchor_words": "None",
                "precision": 0.0
            })

    logging.info("Anchor explanations generated for all recommendations.")
    return json.dumps(explanations)
