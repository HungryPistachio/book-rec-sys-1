import lime
import lime.lime_tabular
import shap
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature names for the Boston Housing dataset
feature_names = boston.feature_names

def get_lime_explanation(model, input_data):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=feature_names)
    exp = explainer.explain_instance(input_data, model.predict)
    return exp.as_list()

def get_shap_explanation(model, input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Combine feature names and SHAP values
    shap_explanation = []
    for i, feature_value in enumerate(input_data):
        shap_explanation.append((feature_names[i], shap_values[i]))
    
    return shap_explanation

def get_counterfactual_explanation(model, input_data, target_value):

    # Implement the counterfactual explanation logic here
    # This is a simplified example, you may need to use more advanced techniques
    
    current_prediction = model.predict([input_data])[0]
    counterfactual_explanation = []
    
    for i, feature_value in enumerate(input_data):
        new_input_data = input_data.copy()
        
        # Iterate through each feature and find the minimum change required to reach the target value
        if current_prediction < target_value:
            new_input_data[i] += 1
        else:
            new_input_data[i] -= 1
        
        new_prediction = model.predict([new_input_data])[0]
        
        if abs(new_prediction - target_value) < abs(current_prediction - target_value):
            counterfactual_explanation.append((feature_names[i], new_input_data[i] - feature_value))
    
    return counterfactual_explanation
