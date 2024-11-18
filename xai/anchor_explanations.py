def get_anchor_explanation_for_recommendation(recommendation, original_feature_names):
    feature_names = recommendation.get("feature_names", [])
    description_vector = recommendation.get("vectorized_descriptions", [])

    # Use top 10 features for input text
    top_features = get_top_features(feature_names, description_vector, top_n=10)
    input_text = ' '.join(top_features)
    logging.info(f"Input text for Anchor explanation: {input_text}")

    if not input_text:
        logging.warning(f"No significant features for recommendation: {recommendation.get('title', 'Unknown')}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=get_spacy_model(), predictor=meaningful_predictor)

    try:
        explanation = explainer.explain(
            input_text,
            threshold=0.1,  # Adjusted threshold
            beam_size=25,   # Increased beam size
            sample_proba=0.7
        )

        # Check if anchors exist
        anchor_words = explanation.data.get('anchor', [])
        if not anchor_words:
            logging.warning(f"No anchors generated for recommendation: {recommendation.get('title', 'Unknown')}")
            return {
                "title": recommendation.get("title", "Recommendation"),
                "anchor_words": "None",
                "precision": 0.0
            }

        # Join anchors and return valid precision
        anchor_words = " and ".join(anchor_words)
        precision = float(explanation.data.get('precision', 0.0))
        logging.info(f"Generated explanation with precision: {precision}, anchors: {anchor_words}")

        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": anchor_words,
            "precision": precision
        }

    except Exception as e:
        logging.error(f"Error generating Anchor explanation: {e}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }
