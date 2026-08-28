from catboost import CatBoostClassifier, Pool
import pandas as pd
import joblib

def fit():
    candidates_for_train = pd.read_parquet('data/candidates_for_train.parquet')
    candidates_to_rank = pd.read_parquet('data/candidates_to_rank.parquet')

    # задаём имена колонок признаков и таргета
    features = ['als_score', 'user_item_views_count', 'user_views_count', 'item_views_count', 'user_addtocart_count',
                'item_addtocart_count', 'user_events_count', 'item_events_count', 'category', 'general_category']
    cat_features = ['category', 'general_category']
    target = 'target'

    # создаём Pool
    train_data = Pool(
        data=candidates_for_train[features],
        cat_features=cat_features,
        label=candidates_for_train[target])

    # инициализируем модель CatBoostClassifier
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=100,
        random_seed=0,
    )

    # тренируем модель
    cb_model.fit(train_data)

    joblib.dump(cb_model, 'models/cb_model.pkl')

    # предсказание
    inference_data = Pool(
    data=candidates_to_rank[features],
    cat_features=cat_features
    )
    predictions = cb_model.predict_proba(inference_data)

    candidates_to_rank["cb_score"] = predictions[:, 1]

    # для каждого пользователя проставим rank, начиная с 1 — это максимальный cb_score
    candidates_to_rank = candidates_to_rank.sort_values(["user_id", "cb_score"], ascending=[True, False])
    candidates_to_rank["rank"] = candidates_to_rank.groupby("user_id").cumcount() + 1

    max_recommendations_per_user = 100
    final_recommendations = candidates_to_rank.query("rank <= @max_recommendations_per_user")

    final_recommendations.to_parquet('data/recommendations.parquet')