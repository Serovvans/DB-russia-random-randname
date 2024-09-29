import pandas as pd
import numpy as np

from src.prepare_events import preppare_events
from src.extract_features import extract_features
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

def get_data():
    """Готовит данные

    Returns:
        (pd.DataFrame, list): Фичи данных и список категориальных столбцов
    """
    preppare_events("train_data/", "train") # Подготавливаем разметку даты
    
    X, cat_features = extract_features("train_data/", "train") # Чистим данные и считаем фичи
    
    return X, cat_features

def train_models():
    """Учит модель
    """
    features, cat_features = get_data()
    X = features.drop(columns=['sex', 'age_class'])

    y1 = features['sex']
    y2 = features['age_class']
    
    sex_decode = {"female": 0, "male": 1}
    sex_encode = {0: "female", 1: "male"}

    y1 = y1.apply(lambda x: sex_decode[x])
    
    X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=["viewer_uid"]), y1, test_size=0.2, random_state=42, stratify=y1)
    param_grid = {
        'iterations': [100, 250, 500],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 8]
    }

    grid_search = GridSearchCV(estimator=CatBoostClassifier(random_state=42, verbose=0,
                                                            cat_features=cat_features),
                            param_grid=param_grid, cv=5, verbose=0)

    grid_search.fit(X_train, y_train)
    best_catboost_model = grid_search.best_estimator_
    
    y1_pred = best_catboost_model.predict(X_test)
    accuracy = accuracy_score(y_test, y1_pred)
    print(f'Acc sex of CatBoost: {accuracy:.2f}')
    
    X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X.drop(columns=['viewer_uid']), y2, test_size=0.2, random_state=42, stratify=y2)
    classes_age = np.unique(y_train_age)
    weights_age = compute_class_weight(class_weight='balanced', classes=classes_age, y=y_train_age)
    class_weights_age = dict(zip(classes_age, weights_age))
    
    param_grid = {
        'iterations': [100, 250],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 10]
    }

    grid_search = GridSearchCV(estimator=CatBoostClassifier(random_state=42, verbose=0,
                                                            cat_features=cat_features,
                                                            class_weights=class_weights_age),
                            param_grid=param_grid, cv=5, verbose=0)

    grid_search.fit(X_train_age, y_train_age)
    best_catboost_model_age = grid_search.best_estimator_
    
    y2_pred = best_catboost_model_age.predict(X_test_age)
    y2_pred = [i[0] for i in y2_pred]
    y_test_age = y_test_age.to_list()
    
    f1 = f1_score(y_test_age, y2_pred, average='weighted')
    print(f'F1 age of CatBoost: {f1:.2f}')
    
    final_score = 0.7 * f1 + 0.3 * accuracy
    print("Finall: ", final_score)
    
    # Сохранение модели пола
    best_catboost_model.save_model('best_catboost_model_sex.cbm')

    # Сохранение модели возраста
    best_catboost_model_age.save_model('best_catboost_model_age.cbm')
    