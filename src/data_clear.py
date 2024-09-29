import pandas as pd

def clear_data(path, part):
    """Чистит данные по указанному пути и части

    Args:
        path (_type_): путь к файлас
        part (_type_): часть (train или test)
    """
    data = pd.read_csv(path + f'{part}_events_times.csv')
    old = pd.read_csv(path + f'{part}_events.csv')
    video = pd.read_csv(path + f'{part}_info_v2.csv')
    if part == "train":
        targets = pd.read_csv(path + f'{part}_targets.csv')
    old = old[['viewer_uid', 'event_timestamp', 'rutube_video_id']]
    
    merge_data = data.merge(video, how="left", on="rutube_video_id")

    if part == 'train':
        merge_data = merge_data.merge(targets[['age', 'age_class', 'sex', 'viewer_uid']],
                                how="left", on="viewer_uid")
        # Очистка выбросов
        merge_data = remove_outliers(merge_data, ['total_watchtime'], method='iqr')

        # Заполнение пропусков
        mode_value = merge_data['ua_os'].mode()[0]
        merge_data['ua_os'].fillna(mode_value, inplace=True)

        mode_value = merge_data['ua_client_name'].mode()[0]
        merge_data['ua_client_name'].fillna(mode_value, inplace=True)


        # Удаление канала, который только зашумляет наши данные
        merge_data = merge_data[merge_data['author_id'] != 1009257]
        merge_data = merge_data[merge_data['author_id'] != 1043618]
        
    if part == "train":
        old['date'] = pd.to_datetime(old['event_timestamp']).dt.day.astype(str)
    else:
        old['date'] = pd.to_datetime(old['event_timestamp']).dt.date.apply(str).astype(str)

    old = old.drop(columns=['event_timestamp'])

    merge_data['date'] = old['date'][merge_data.index]
    
    return merge_data

    
def remove_outliers(df, columns, method='iqr', factor=1.5):
    """
    Удаляет выбросы из указанных столбцов DataFrame с использованием указанного метода.

    Параметры:
    - df (pd.DataFrame): исходный DataFrame
    - columns (list): список столбцов для очистки от выбросов
    - method (str): метод для удаления выбросов, может быть 'iqr' или 'std'
                    ('iqr' - интерквартильный размах, 'std' - стандартное отклонение)
    - factor (float): множитель для определения границ выбросов (по умолчанию 1.5 для IQR и 3 для std)

    Возвращает:
    - pd.DataFrame: DataFrame без выбросов в указанных столбцах
    """

    df_clean = df.copy()

    if method not in ['iqr', 'std']:
        raise ValueError("Метод должен быть 'iqr' или 'std'")

    for column in columns:
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

        elif method == 'std':
            mean = df_clean[column].mean()
            std_dev = df_clean[column].std()
            lower_bound = mean - factor * std_dev
            upper_bound = mean + factor * std_dev

        # Фильтрация DataFrame по границам выбросов
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]

    return df_clean