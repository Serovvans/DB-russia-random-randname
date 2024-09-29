# Предсказание социально-демографических характеристик пользователя

Проект созда в рамках решения кейса Всероссийского хакатно Цифровой прорыв

## Установка

### Используя conda

1. Установите [Anaconda](https://www.anaconda.com/products/distribution) или [Miniconda](https://docs.conda.io/en/latest/miniconda.html), если она ещё не установлена.

2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Serovvans/DB-russia-random-randname.git
   cd DB-russia-random-randname
   ```

3. Создайте окружение и установите зависимости:
    ```bash
    conda create --name myenv --file requirements.txt
    ```

4. Активируйте окружение
    ```bash
    conda activate myenv
    ```

## Запуск проекта

1. Для обучения положите тренировочный датасет в папку train_data, созданную от корня проекта и выполните команду:

    ```bash
    python train.py
    ```

1. Для предсказания положите тестовый датасет в папку test_data, созданную от корня проекта и выполните команду:

    ```bash
    python predict_test.py
    ```

## Структура проекта

```bash
├── train_data/                     # Тренировочный датасет
│              # Тренировочный датасет (train_events, train_targets, video_info_v2)
├── notebooks/                # Jupyter Notebooks
├── src/                      # Исходный код проекта
│   ├── __init__.py           # Инициализация пакета
│   ├── data_clear.py    # Скрипты для обработки данных
│   ├── extract_features.py              # Скрипты для создания и обучения модели
│   └── prepare_events.py              # Утилитарные функции
├── train_data/                    # Тестовый датасет
│         # Тестирование модели
├── requirements.txt          # Файл с зависимостями для pip
├── README.md                 # Описание проекта
├── predict_test.py           # Главный скрипт для создания предсказания
└── train.py                   # Главный скрипт для тренировки модели
```
