import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

df = pd.read_excel("./sample_data/data.xlsx", 'Лист1')

df = df.head(100)

# 2. Подготовка данных
# Список колонок, которые нужно соединить в одну строку для анализа
columns_to_join = ['название сте', 'модель', 'производитель', 'название категории', 'характеристики']
#columns_to_join = ['название сте']

# Проверяем, есть ли эти колонки в файле, чтобы избежать ошибок
missing_cols = [col for col in columns_to_join if col not in df.columns]
if missing_cols:
    raise ValueError(f"В файле не найдены колонки: {missing_cols}")

# Заполняем пустые значения, приводим всё к строке и объединяем через пробел
df['combined_text'] = df[columns_to_join].fillna('').astype(str).agg(' '.join, axis=1)

# Очистка от лишних пробелов (опционально, но полезно)
df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True).str.strip()


# 3. Векторизация
# Используем SBERT для русского языка (лучше всего подходит для Entity Resolution)
model = SentenceTransformer("ai-forever/FRIDA")

embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

# Нормализация векторов (важно для корректной работы косинусного расстояния)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 4. Кластеризация (Entity Resolution)
# AgglomerativeClustering отлично подходит, когда мы не знаем число кластеров,
# но знаем, насколько похожими должны быть дубликаты.

# distance_threshold - порог схожести (0.0 - полное совпадение, 1.0 - полная разница).
# 0.15 - 0.25 обычно хороший диапазон для SBERT.
threshold = 0.1

print(f"Запуск кластеризации с порогом {threshold}...")
cluster_model = AgglomerativeClustering(
    n_clusters=None,             # Автоматическое определение количества групп
    distance_threshold=threshold,
    metric='cosine',             # Используем косинусное расстояние
    linkage='average'            # Метод связывания average дает стабильные результаты
)

# Присваиваем каждому товару ID его группы (кластера)
df['group_id'] = cluster_model.fit_predict(embeddings)

# 5. Сохранение результата
output_file = 'result_with_groups.xlsx'

# Сортируем по group_id, чтобы дубликаты шли друг за другом в Excel
df_sorted = df.sort_values(by='group_id')

df_sorted.to_excel(output_file, index=False)

# Вывод небольшого примера в консоль
print("\nПример найденных групп:")
example_groups = df_sorted['group_id'].value_counts().head(3).index # Берем топ-3 крупных кластера
print(df_sorted[df_sorted['group_id'].isin(example_groups)][['group_id', 'combined_text']].head(10))