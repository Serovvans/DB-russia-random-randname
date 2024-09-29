import pandas as pd

from catboost import CatBoostClassifier
from src.prepare_events import preppare_events
from src.extract_features import extract_features

def get_data():
    """Готовит данные

    Returns:
        (pd.DataFrame, list): Фичи данных и список категориальных столбцов
    """
    preppare_events("test_data/", "test") # Подготавливаем разметку даты
    
    X, cat_features = extract_features("test_data/", "test") # Чистим данные и считаем фичи
    
    return X, cat_features

def test():
    """Создает предикт для теста
    """
    features, cat_features = get_data()
    X = features
    
    # Загрузка моделей
    best_catboost_model = CatBoostClassifier(cat_features=cat_features)
    best_catboost_model_age = CatBoostClassifier(cat_features=cat_features)
    
    best_catboost_model.load_model("best_catboost_model_sex.cbm")
    best_catboost_model_age.load_model("best_catboost_model_age.cbm")
    
    # Формируем предсказание
    y1_pred = best_catboost_model.predict(X.drop(columns=['viewer_uid']))
    y2_pred = best_catboost_model_age.predict(X.drop(columns=['viewer_uid']))
    y2_pred = [i[0] for i in y2_pred]
    decode_age_class = {0: 15, 1: 25, 2: 35, 3: 50}
    age = [decode_age_class[clas] for clas in y2_pred]
    
    sex_encode = {0: "female", 1: "male"}

    y1_pred = [sex_encode[i] for i in y1_pred]
    
    # Сохраняем предсказание
    answer = pd.DataFrame({"age": age, "sex": y1_pred, "age_class": y2_pred}, index=X['viewer_uid'])
    answer.to_csv("submission.csv")
    
if __name__ == "__main__":
    test()
