import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def cluster_entities(
        input_file: str = "./sample_data/data.xlsx",
        sheet_name: str = "Лист1",
        columns_to_join: list = None,
        n_samples: int = 100,
        distance_threshold: float = 0.1,
        output_file: str = "result_with_groups.xlsx",
        embedding_model_name: str = "ai-forever/FRIDA"
):
    """
    Кластеризует записи из Excel-файла на основе семантического сходства текстовых полей.

    Parameters:
    -----------
    input_file : str
        Путь к исходному Excel-файлу.
    sheet_name : str
        Имя листа в Excel-файле.
    columns_to_join : list of str
        Список колонок, которые будут объединены в одну текстовую строку для анализа.
        Если None, используется дефолтный список.
    n_samples : int
        Количество строк из начала файла для обработки (для тестирования/демонстрации).
    distance_threshold : float
        Порог косинусного расстояния для агglomerативной кластеризации.
        Меньше значение — строже критерий похожести.
    output_file : str
        Путь к выходному Excel-файлу с результатами.
    embedding_model_name : str
        Название модели SentenceTransformer для векторизации текста.

    Returns:
    --------
    pd.DataFrame
        DataFrame с добавленной колонкой 'group_id'.
    """
    if columns_to_join is None:
        columns_to_join = ['название сте', 'модель', 'производитель', 'название категории', 'характеристики']

    # 1. Загрузка данных
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    df = df.head(n_samples)

    # 2. Подготовка данных
    missing_cols = [col for col in columns_to_join if col not in df.columns]
    if missing_cols:
        raise ValueError(f"В файле не найдены колонки: {missing_cols}")

    df['combined_text'] = df[columns_to_join].fillna('').astype(str).agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 3. Векторизация
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 4. Кластеризация
    print(f"Запуск кластеризации с порогом {distance_threshold}...")
    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    df['group_id'] = cluster_model.fit_predict(embeddings)

    # 5. Сохранение результата
    df_sorted = df.sort_values(by='group_id')
    df_sorted.to_excel(output_file, index=False)

    # Вывод примера
    print("\nПример найденных групп:")
    example_groups = df_sorted['group_id'].value_counts().head(3).index
    print(df_sorted[df_sorted['group_id'].isin(example_groups)][['group_id', 'combined_text']].head(10))

    return df_sorted


cluster_entities(
    input_file="./sample_data/data.xlsx",
    sheet_name="Лист1",
    columns_to_join=['название сте', 'модель', 'производитель', 'название категории', 'характеристики'],
    n_samples=200,
    distance_threshold=0.15,
    output_file="my_clusters.xlsx"
)
