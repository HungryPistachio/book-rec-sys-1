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
            "anchor_words": "No significant anchors identified",
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
        precision = float(explanation.data.get('precision', 0.0))

        # Handle cases with no anchors
        if not anchor_words:
            logging.warning(f"No anchors generated for recommendation: {recommendation.get('title', 'Unknown')}")
            anchor_words = "No significant anchors identified"
            precision = 0.0

        logging.info(f"Generated explanation with precision: {precision}, anchors: {anchor_words}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": " AND ".join(anchor_words) if anchor_words else "No significant anchors identified",
            "precision": precision
        }
    except Exception as e:
        logging.error(f"Error generating Anchor explanation: {e}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "No significant anchors identified",
            "precision": 0.0
        }
