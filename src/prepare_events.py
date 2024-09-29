import pandas as pd
from datetime import timedelta

def preppare_events(path, part):
    """Подготавливает временные данные в информации о просмотрах

    Args:
        path (str): путь к файлас
        part (str): трейн или тест

    """
    data = pd.read_csv(path+f"{part}_events.csv")
    
    data['time'] = pd.to_datetime(data['event_timestamp']).dt.time.apply(str)

    # Словарь со смещениями для каждого региона относительно Московского времени
    timezone_offsets = {
        'Moscow': timedelta(hours=0),
        'Chelyabinsk': timedelta(hours=2),
        'Tatarstan Republic': timedelta(hours=0),
        'Bashkortostan Republic': timedelta(hours=2),
        'Novosibirsk Oblast': timedelta(hours=4),
        'Moscow Oblast': timedelta(hours=0),
        'Omsk Oblast': timedelta(hours=3),
        'Chuvashia': timedelta(hours=0),
        'Krasnoyarsk Krai': timedelta(hours=4),
        'Kamchatka': timedelta(hours=9),
        'Nizhny Novgorod Oblast': timedelta(hours=0),
        'Krasnodar Krai': timedelta(hours=0),
        'Volgograd Oblast': timedelta(hours=1),
        'Kaliningrad Oblast': timedelta(hours=-1),
        'Kuzbass': timedelta(hours=4),
        'Stavropol Kray': timedelta(hours=0),
        'Samara Oblast': timedelta(hours=1),
        'Amur Oblast': timedelta(hours=6),
        'Sverdlovsk Oblast': timedelta(hours=2),
        'St.-Petersburg': timedelta(hours=0),
        'Yamalo-Nenets': timedelta(hours=2),
        'Orenburg Oblast': timedelta(hours=2),
        'Khanty-Mansia': timedelta(hours=2),
        'Kaluga Oblast': timedelta(hours=0),
        'Tomsk Oblast': timedelta(hours=4),
        'Novgorod Oblast': timedelta(hours=0),
        'Arkhangelskaya': timedelta(hours=0),
        'North Ossetia–Alania': timedelta(hours=0),
        'Kursk Oblast': timedelta(hours=0),
        'Leningradskaya Oblast': timedelta(hours=0),
        'Krasnoyarskiy': timedelta(hours=4),
        'Ivanovo Oblast': timedelta(hours=0),
        'Altay Kray': timedelta(hours=4),
        'Kurgan Oblast': timedelta(hours=2),
        'Kostroma Oblast': timedelta(hours=0),
        'Bryansk Oblast': timedelta(hours=0),
        'Dagestan': timedelta(hours=0),
        'Lipetsk Oblast': timedelta(hours=0),
        'Vladimir Oblast': timedelta(hours=0),
        'Kirov Oblast': timedelta(hours=0),
        'Khabarovsk': timedelta(hours=7),
        'Tambov Oblast': timedelta(hours=0),
        'Chukotka': timedelta(hours=9),
        'Voronezh Oblast': timedelta(hours=0),
        'Sverdlovsk': timedelta(hours=2),
        'Tula Oblast': timedelta(hours=0),
        'Krasnodarskiy': timedelta(hours=0),
        'Irkutsk Oblast': timedelta(hours=5),
        'Saratov Oblast': timedelta(hours=1),
        'Khakasiya Republic': timedelta(hours=4),
        'Penza': timedelta(hours=0),
        'Perm Krai': timedelta(hours=2),
        'Oryol oblast': timedelta(hours=0),
        'Vladimir': timedelta(hours=0),
        'Smolensk Oblast': timedelta(hours=0),
        'Penza Oblast': timedelta(hours=0),
        'Mordoviya Republic': timedelta(hours=0),
        'Tyumen’ Oblast': timedelta(hours=2),
        'Sakha': timedelta(hours=6),
        'Primorye': timedelta(hours=7),
        'Zabaykalskiy (Transbaikal) Kray': timedelta(hours=6),
        'Vologda Oblast': timedelta(hours=0),
        'Yaroslavl Oblast': timedelta(hours=0),
        'Crimea': timedelta(hours=0),
        'Rostov': timedelta(hours=0),
        'Ryazan Oblast': timedelta(hours=0),
        'Perm': timedelta(hours=2),
        'Chechnya': timedelta(hours=0),
        'Udmurtiya Republic': timedelta(hours=1),
        'Tver Oblast': timedelta(hours=0),
        'Buryatiya Republic': timedelta(hours=5),
        'Belgorod Oblast': timedelta(hours=0),
        'Kaluga': timedelta(hours=0),
        'Astrakhan Oblast': timedelta(hours=1),
        'Karelia': timedelta(hours=0),
        'Murmansk': timedelta(hours=0),
        'Adygeya Republic': timedelta(hours=0),
        'Kemerovo Oblast': timedelta(hours=4),
        'Mariy-El Republic': timedelta(hours=0),
        'Kursk': timedelta(hours=0),
        'Saratovskaya Oblast': timedelta(hours=1),
        'Sakhalin Oblast': timedelta(hours=8),
        'Ivanovo': timedelta(hours=0),
        'Tyumen Oblast': timedelta(hours=2),
        'Stavropol’ Kray': timedelta(hours=0),
        'Voronezj': timedelta(hours=0),
        'Karachayevo-Cherkesiya Republic': timedelta(hours=0),
        'Kabardino-Balkariya Republic': timedelta(hours=0),
        'Ulyanovsk': timedelta(hours=1),
        'North Ossetia': timedelta(hours=0),
        'Komi': timedelta(hours=0),
        'Smolensk': timedelta(hours=0),
        'Tver’ Oblast': timedelta(hours=0),
        'Sebastopol City': timedelta(hours=0),
        'Pskov Oblast': timedelta(hours=0),
        'Tula': timedelta(hours=0),
        'Orel Oblast': timedelta(hours=0),
        'Jaroslavl': timedelta(hours=0),
        'Tambov': timedelta(hours=0),
        'Kalmykiya Republic': timedelta(hours=0),
        'Primorskiy (Maritime) Kray': timedelta(hours=7),
        'Altai': timedelta(hours=4),
        'Magadan Oblast': timedelta(hours=8),
        'Vologda': timedelta(hours=0),
        'Tyva Republic': timedelta(hours=4),
        'Nenets': timedelta(hours=0),
        'Smolenskaya Oblast’': timedelta(hours=0),
        'Jewish Autonomous Oblast': timedelta(hours=7),
        'Astrakhan': timedelta(hours=1),
        'Ingushetiya Republic': timedelta(hours=0),
        'Kirov': timedelta(hours=0),
        'Transbaikal Territory': timedelta(hours=5),
        'Omsk': timedelta(hours=3),
        'Kaliningrad': timedelta(hours=-1),
        'Stavropol Krai': timedelta(hours=0),
        'Arkhangelsk Oblast': timedelta(hours=0)
    }

    # Преобразование столбца time в формат timedelta, чтобы работать только с временем
    data['time'] = pd.to_timedelta(data['time'])

    # Функция для изменения времени на основе региона
    def adjust_time(row):
        region = row['region']
        if region in timezone_offsets:
            return row['time'] + timezone_offsets[region]
        else:
            return row['time']  # Если регион не найден в словаре, время не изменяется

    # Применение функции ко всему DataFrame
    data['adjusted_time'] = data.apply(adjust_time, axis=1)

    # Преобразование timedelta обратно в формат 'HH:MM:SS', удаляя '0 days'
    data['time'] = data['time'].apply(lambda x: (pd.Timestamp(0) + x).strftime('%H:%M:%S'))
    data['adjusted_time'] = data['adjusted_time'].apply(lambda x: (pd.Timestamp(0) + x).strftime('%H:%M:%S'))

    # Сохранение результата в новый CSV-файл
    data.to_csv(path+f"{part}_events_times.csv", index=False)