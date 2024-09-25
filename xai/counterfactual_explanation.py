import dice_ml
from dice_ml import Dice
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

def get_counterfactual_explanation(book_title, book_description, all_descriptions):
    # Vectorize the descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    # Create a simple logistic regression model to explain (could use any model)
    model = LogisticRegression()
    labels = [1 if "mystery" in desc else 0 for desc in all_descriptions]  # Example label assignment
    model.fit(tfidf_matrix, labels)

    # Create a DiCE explainer
    d = dice_ml.Data(dataframe=pd.DataFrame(tfidf_matrix.toarray()), continuous_features=[], outcome_name='label')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(d, m)

    # Generate a counterfactual explanation for the input book description
    query_instance = pd.DataFrame(vectorizer.transform([book_description]).toarray())
    counterfactual = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")

    # General explanation
    general_explanation = (
        f"Counterfactuals show how changing certain words would alter the recommendation for '{book_title}'. "
        f"If the word was present or absent, the recommendation would be different."
    )
    
    return {
        "title": book_title,
        "general_explanation": general_explanation,    # General explanation once
        "counterfactual_output": counterfactual.cf_examples_list[0].final_cfs_df.columns.tolist()  # Raw counterfactual output
    }
