import nltk
import pandas as pd

from src.data_clear import clear_data
from nltk.corpus import stopwords
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from collections import Counter
from sklearn.impute import SimpleImputer
from datetime import datetime

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Инициализация инструментов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def preprocess_text(text):
    """Очищает и нормализует тексты заголовков

    Args:
        text (str): тексты

    Returns:
        List[str]: список извлеченных существительных
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    # Приведение к нормальной форме и фильтрация
    normalized_words = []
    for token in doc.tokens:
        if token.text.lower() not in russian_stopwords and token.pos == 'NOUN':  # Например, оставляем только существительные
            token.lemmatize(morph_vocab)
            normalized_words.append(token.lemma)
    return normalized_words


def extract_features(path, part):
    """Чистит данные и считает фичи

    Args:
        path (str): путь к файлас
        part (str): тест или трейн

    Returns:
        (pd.DataFrame, list): Датасет с посчитаными признаками и список категориальных признаков
    """
    
    data = clear_data(path, part)
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
        
    data['cleaned_title'] = data['title'].apply(preprocess_text)
    user_titles = data.groupby('viewer_uid')['cleaned_title'].apply(lambda x: [word for sublist in x for word in sublist]).reset_index()

    def most_common_words(words_list):
        counter = Counter(words_list)
        return [word for word, _ in counter.most_common(5)]

    user_titles['top_5_words'] = user_titles['cleaned_title'].apply(most_common_words)
    for i in range(5):
        user_titles[f'top_word_{i+1}'] = user_titles['top_5_words'].apply(lambda x: x[i] if len(x) > i else 'none')
        
    # Среднее время просмотра на пользователя
    user_watchtime = data.groupby('viewer_uid')['total_watchtime'].mean().reset_index()
    user_watchtime.columns = ['viewer_uid', 'avg_watchtime']

    # Общее количество просмотров для каждого пользователя
    total_views = data.groupby('viewer_uid')['rutube_video_id'].count().reset_index()
    total_views.columns = ['viewer_uid', 'total_views']

    # Количество уникальных видео для каждого пользователя
    unique_videos = data.groupby('viewer_uid')['rutube_video_id'].nunique().reset_index()
    unique_videos.columns = ['viewer_uid', 'unique_videos']

    # Средняя длительность видео
    avg_video_duration = data.groupby('viewer_uid')['duration'].mean().reset_index()
    avg_video_duration.columns = ['viewer_uid', 'avg_video_duration']

    # Частота категорий видео
    top_category = data.groupby('viewer_uid')['category'].agg(lambda x: x.mode()[0]).reset_index()
    top_category.columns = ['viewer_uid', 'top_category']

    data['adjusted_time'] = pd.to_datetime(data['adjusted_time'])

    # Извлечение часа из времени просмотра
    data['hour'] = data['adjusted_time'].dt.hour

    # Категоризация времени на части дня
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
        
    def calculate_weekday(day):
        # 1 июля - это суббота, номер дня в неделе для субботы - 5 (0 - понедельник, 6 - воскресенье)
        # Используем модульное деление для определения дня недели
        if type(day) == type(1):
            start_day_of_week = 5  # Суббота
        else:
            date_obj = datetime.strptime(day, "%Y-%m-%d")
            day_of_week = date_obj.weekday()
            
            return day_of_week
            
        return (start_day_of_week + (day - 1)) % 7

    data['time_of_day'] = data['hour'].apply(get_time_of_day)

    most_cummon_time_of_day = data.groupby('viewer_uid')['time_of_day'].agg(lambda x: x.mode()[0]).reset_index()
    most_cummon_time_of_day.columns = ['viewer_uid', 'most_cummon_time_of_day']

    data['weekday'] = data['date'].apply(calculate_weekday)
    data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # День недели, в который чаще всего смотрят видео
    most_common_weekday = data.groupby('viewer_uid')['weekday'].agg(lambda x: x.mode()[0]).reset_index()
    most_common_weekday.columns = ['viewer_uid', 'most_common_weekday']

    # Среднее количество просмотров одного и того же автора для каждого пользователя
    author_views = data.groupby(['viewer_uid', 'author_id'])['rutube_video_id'].count().reset_index()
    author_views.columns = ['viewer_uid', 'author_id', 'views_per_author']
    avg_views_per_author = author_views.groupby('viewer_uid')['views_per_author'].mean().reset_index()
    avg_views_per_author.columns = ['viewer_uid', 'avg_views_per_author']

    # Получаем 3 самых популярных авторов для каждого пользователя
    def top_authors(series):
        top_authors_list = series.value_counts().nlargest(3).index.tolist()
        # Если авторов меньше 3, заполняем NaN
        return top_authors_list + [None] * (3 - len(top_authors_list))

    top_authors = data.groupby('viewer_uid')['author_id'].agg(top_authors).reset_index()

    # Преобразуем список из 3 авторов в 3 отдельных столбца
    top_authors[['top_author_1', 'top_author_2', 'top_author_3']] = pd.DataFrame(top_authors['author_id'].tolist(), index=top_authors.index)

    # Убираем исходный столбец со списком авторов
    top_authors = top_authors.drop(columns=['author_id'])

    # Концентрация по категориям
    category_concentration = data.groupby('viewer_uid')['category'].agg(lambda x: x.value_counts(normalize=True).max()).reset_index()
    category_concentration.columns = ['viewer_uid', 'category_concentration']

    # Разница во времени между просмотрами для каждого пользователя
    data['watch_time_diff'] = data.groupby('viewer_uid')['adjusted_time'].diff().dt.total_seconds()
    avg_time_between_views = data.groupby('viewer_uid')['watch_time_diff'].mean().reset_index()
    avg_time_between_views.columns = ['viewer_uid', 'avg_time_between_views']

    # Доля просмотров в выходные для каждого пользователя
    weekend_views_ratio = data.groupby('viewer_uid')['is_weekend'].mean().reset_index()
    weekend_views_ratio.columns = ['viewer_uid', 'weekend_views_ratio']

    # Объединение всех признаков в одну таблицу
    features = data[['viewer_uid', 'region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name']].drop_duplicates(subset=['viewer_uid'])

    features = features.merge(user_watchtime, on='viewer_uid', how='left')
    features = features.merge(avg_video_duration, on='viewer_uid', how='left')
    features = features.merge(top_category, on='viewer_uid', how='left')
    features = features.merge(category_concentration, on='viewer_uid', how='left')
    features = features.merge(most_cummon_time_of_day, on='viewer_uid', how='left')
    features = features.merge(most_common_weekday, on='viewer_uid', how='left')
    features = features.merge(avg_views_per_author, on='viewer_uid', how='left')
    features = features.merge(top_authors, on='viewer_uid', how='left')
    features = features.merge(avg_time_between_views, on='viewer_uid', how='left')
    features = features.merge(weekend_views_ratio, on='viewer_uid', how='left')
    features = features.merge(total_views, on='viewer_uid', how='left')
    features = features.merge(user_titles[['viewer_uid', 'top_word_1', 'top_word_2', 'top_word_3', 'top_word_4', 'top_word_5']], on='viewer_uid', how='left')

    features['rel_watchtime'] = features['avg_watchtime'] / features['avg_video_duration']
    features = features.drop(columns=['avg_watchtime', 'ua_client_type', 'avg_video_duration'])

    if part == "train":
    # Добавляем пол
        features = features.merge(data[['viewer_uid', 'sex', 'age_class']].drop_duplicates(subset=['viewer_uid']), on='viewer_uid', how='left')

    # Заполнение пропусков
    imputer = SimpleImputer(strategy='most_frequent')
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    for col in ['top_author_1', 'top_author_2', 'top_author_3']:
        features[col] = features[col].apply(int).astype(int)

    cat_features = ['region', 'ua_device_type', 'ua_os', 'ua_client_name', 'top_category', 'most_cummon_time_of_day', 'most_common_weekday',
                    'top_author_1', 'top_author_2', 'top_author_3', 'top_word_1', 'top_word_2', 'top_word_3', 'top_word_4', 'top_word_5']
            
    for col in cat_features:
        features[col] = features[col].astype(str)
    

    X = features
    
    return X, cat_features
        