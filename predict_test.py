import pandas as pd

from catboost import CatBoostClassifier
from src.prepare_events import preppare_events
from src.extract_features import extract_features

def get_data():
    preppare_events("test_data/", "test")
    
    X, cat_features = extract_features("test_data/", "test")
    
    return X, cat_features

def test():
    features, cat_features = get_data()
    X = features
    
    best_catboost_model = CatBoostClassifier()
    best_catboost_model_age = CatBoostClassifier
    
    best_catboost_model.load_model("best_catboost_model_sex.cbm")
    best_catboost_model_age.load_model("best_catboost_model_age.cbm")
    
    y1_pred = best_catboost_model.predict(X)
    y2_pred = best_catboost_model_age.predict(X)
    y2_pred = [i[0] for i in y2_pred]
    
    answer = pd.DateFrame({"id": X.index, "sex": y1_pred, "age_class": y2_pred})
    answer.to_csv("submission.csv")
    
if __name__ == "__main__":
    test()
