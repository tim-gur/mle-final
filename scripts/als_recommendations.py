import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import scipy
import numpy as np
from implicit.als import AlternatingLeastSquares

def als_recommendations():
    events_train = pd.read_parquet('data/events_train.parquet')
    events_test = pd.read_parquet('data/events_test.parquet')
    events = pd.read_parquet('data/events.parquet')

    # кодирование юзеров и айтемов для матрицы
    user_encoder = LabelEncoder()
    user_encoder.fit(events['user_id'])
    events_train["user_id_enc"] = user_encoder.transform(events_train["user_id"])
    events_test["user_id_enc"] = user_encoder.transform(events_test["user_id"])

    item_encoder = LabelEncoder()
    item_encoder.fit(events['item_id'])
    events_train['item_id_enc'] = item_encoder.transform(events_train['item_id'])
    events_test['item_id_enc'] = item_encoder.transform(events_test['item_id'])

    # создание матрицы
    events_train_view_addtocart = events_train[events_train['event'].isin(['view', 'addtocart'])].copy()

    weight_map = {'view': 0.3, 'addtocart': 1.0}
    events_train_view_addtocart['weight'] = events_train_view_addtocart['event'].map(weight_map)
    interactions_train = (events_train_view_addtocart.groupby(['user_id_enc', 'item_id_enc'])['weight'].max().reset_index())

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)

    user_item_matrix_train = scipy.sparse.csr_matrix(
        (interactions_train['weight'],
        (interactions_train['user_id_enc'], interactions_train['item_id_enc'])),
        shape=(n_users, n_items),
        dtype=np.float32
    )

    # построение модели
    als_model = AlternatingLeastSquares(factors=50, iterations=50, regularization=0.05, random_state=42)
    als_model.fit(user_item_matrix_train)

    # получаем список всех возможных user_id (перекодированных)
    user_ids_encoded = range(len(user_encoder.classes_))

    # получаем рекомендации для всех пользователей
    als_recommendations = als_model.recommend(
        user_ids_encoded, 
        user_item_matrix_train[user_ids_encoded], 
        filter_already_liked_items=False, N=100)

    # преобразуем полученные рекомендации в табличный формат
    item_ids_enc = als_recommendations[0]
    als_scores = als_recommendations[1]

    als_recommendations = pd.DataFrame({
        "user_id_enc": user_ids_encoded,
        "item_id_enc": item_ids_enc.tolist(), 
        "score": als_scores.tolist()})
    als_recommendations = als_recommendations.explode(["item_id_enc", "score"], ignore_index=True)

    # приводим типы данных
    als_recommendations["item_id_enc"] = als_recommendations["item_id_enc"].astype("int")
    als_recommendations["score"] = als_recommendations["score"].astype("float")

    # получаем изначальные идентификаторы
    als_recommendations["user_id"] = user_encoder.inverse_transform(als_recommendations["user_id_enc"])
    als_recommendations["item_id"] = item_encoder.inverse_transform(als_recommendations["item_id_enc"])
    als_recommendations = als_recommendations.drop(columns=["user_id_enc", "item_id_enc"])

    als_recommendations.to_parquet('data/als_recommendations.parquet')

if __name__ == '__main__':
    als_recommendations()