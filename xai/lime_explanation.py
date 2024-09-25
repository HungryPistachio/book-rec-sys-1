import lime
import lime.lime_text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_lime_explanation(book_title, book_description, all_descriptions):
    # Vectorize the descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    # Define LIME explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Book Recommendation"])

    # Simulate a prediction function based on cosine similarity
    def predict_fn(texts):
        text_tfidf = vectorizer.transform(texts)
        similarities = np.dot(tfidf_matrix, text_tfidf.T).toarray()
        return similarities.mean(axis=0).reshape(-1, 1)

    # Generate explanation using LIME
    exp = explainer.explain_instance(book_description, predict_fn, num_features=6)
    
    explanation_list = exp.as_list()
    
    # General explanation
    general_explanation = (
        f"LIME provides a feature importance score by measuring how much each word "
        f"influences the similarity between the book '{book_title}' and others. "
        f"A higher weight means the word has a stronger influence."
    )
    
    return {
        "title": book_title,
        "general_explanation": general_explanation, # General explanation once
        "lime_output": explanation_list             # Raw LIME output
    }
