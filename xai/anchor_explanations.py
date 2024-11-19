import json
import logging
import pandas as pd
from alibi.explainers import AnchorText
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global NLP model
nlp = None

# Load vocabulary
vocab_df = pd.read_csv('static/fixed_vocabulary.csv')
vocab = set(vocab_df['Vocabulary'])  # Convert to lowercase for consistency



def get_spacy_model():
    """
    Load spaCy model only when needed to conserve memory.
    """
    global nlp
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    return nlp

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Global original vector for similarity computation
original_vector = None

def prepare_vectorizer_and_original_vector(original_feature_names):
    """
    Prepare the vectorizer and create the original vector.
    """
    global original_vector
    original_description = ' '.join(original_feature_names)
    vectorizer.fit([original_description])
    original_vector = vectorizer.transform([original_description]).toarray()[0]
    return original_vector

def get_top_features(feature_names, description_vector, top_n=20):
    """
    Get the top N features by vector weight.
    """
    feature_importance = list(zip(feature_names, description_vector))
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    return top_features

def meaningful_predictor(texts):
    """
    A predictor function that returns a binary prediction based on similarity,
    ignoring perturbed examples with UNK tokens.
    """
    global original_vector
    clean_texts = [preprocess_text(text, vocab) for text in texts if "UNK" not in text]
    if not clean_texts:
        return np.zeros(len(texts))  # Return 0 for all if no valid examples exist

    text_vectors = vectorizer.transform(clean_texts).toarray()
    similarities = np.dot(text_vectors, original_vector) / (
            np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector)
    )

    # Map predictions back to the original input size
    predictions = (similarities > 0.05).astype(int)

    # Fill skipped examples with default prediction (e.g., 0)
    full_predictions = np.zeros(len(texts))
    valid_idx = [i for i, text in enumerate(texts) if "UNK" not in text]
    for idx, pred in zip(valid_idx, predictions):
        full_predictions[idx] = pred

    return full_predictions

def preprocess_text(text, vocab):
    """
    Preprocess text to remove words not in the vocabulary and exclude single characters.
    """
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if len(word) > 1 and word.lower() in vocab]
    return ' '.join(cleaned_tokens)

def get_anchor_explanation_for_recommendation(recommendation, original_feature_names):
    feature_names = recommendation.get("feature_names", [])
    description_vector = recommendation.get("vectorized_descriptions", [])

    # Use top 10 features for input text
    top_features = get_top_features(feature_names, description_vector, top_n=20)
    input_text = preprocess_text(' '.join(top_features), vocab)
    logging.info(f"Input text for Anchor explanation: {input_text}")

    if not input_text.strip():
        logging.warning(f"No significant features for recommendation: {recommendation.get('title', 'Unknown')}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "0",
            "precision": 0.0
        }

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=get_spacy_model(), predictor=meaningful_predictor)

    try:
        explanation = explainer.explain(
            input_text,
            threshold=0.05,  # Allow smaller coverage
            beam_size=20,    # Increase search space
            sample_proba=0.7 # Balance sampling diversity
        )

        anchor_words = explanation.data.get('anchor', [])
        precision = round(float(explanation.data.get('precision', 0.0)), 4)



        # Handle cases with no anchors
        if not anchor_words:
            logging.warning(f"No anchors generated for recommendation: {recommendation.get('title', 'Unknown')}")
            anchor_words = "0"
            precision = 0.0

        logging.info(f"Generated explanation with precision: {precision}, anchors: {anchor_words}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": " - ".join(anchor_words),
            "precision": precision
        }
    except Exception as e:
        logging.error(f"Error generating Anchor explanation: {e}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "No significant anchors identified",
            "precision": 0.0
        }



def get_anchor_explanation(recommendations, original_feature_names):
    """
    Generate anchor explanations for all recommendations.
    """
    explanations = []

    # Prepare the vectorizer and original vector once
    if isinstance(original_feature_names, str):
        original_feature_names = original_feature_names.split()
    prepare_vectorizer_and_original_vector(original_feature_names)

    for idx, rec in enumerate(recommendations):
        explanation = get_anchor_explanation_for_recommendation(rec, original_feature_names)
        # Convert NumPy types to native Python types for JSON serialization
        explanation = json.loads(json.dumps(explanation, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        explanations.append(explanation)

    return json.dumps(explanations, indent=4)
