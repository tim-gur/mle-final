import pandas as pd
import numpy as np
import os
import json
import mlflow
from dotenv import load_dotenv
import joblib

def evaluate():
    events_train = pd.read_parquet('data/events_train.parquet')
    events_test = pd.read_parquet('data/events_test.parquet')
    final_recommendations = pd.read_parquet('data/recommendations.parquet')
    als_recommendations = pd.read_parquet('data/als_recommendations.parquet')

    # NDCG@10
    def dcg_at_k(ranked_items, relevant_items, k):
        ranked_items = ranked_items[:k]
        dcg = 0.0
        for i, item in enumerate(ranked_items, start=1):
            rel = 1 if item in relevant_items else 0
            dcg += rel / np.log2(i + 1)
        return dcg

    def ndcg_at_k(ranked_items, relevant_items, k):
        ideal_dcg = dcg_at_k(list(relevant_items), relevant_items, k)  # best possible ordering
        if ideal_dcg == 0:
            return 0.0
        return dcg_at_k(ranked_items, relevant_items, k) / ideal_dcg

    def mean_ndcg_at_k(user_rankings, user_relevant, k):
        scores = []
        for user_id, ranked_items in user_rankings.items():
            relevant_items = user_relevant.get(user_id, set())
            scores.append(ndcg_at_k(ranked_items, relevant_items, k))
        return np.mean(scores)

    def process_events_recs_for_binary_metrics(events_train, events_test, recs, top_k=None):

        """
        размечает пары <user_id, item_id> для общего множества пользователей признаками
        - gt (ground truth)
        - pr (prediction)
        top_k: расчёт ведётся только для top k-рекомендаций
        """

        events_test["gt"] = True
        common_users = set(events_test["user_id"]) & set(recs["user_id"])

        print(f"Common users: {len(common_users)}")
        
        events_for_common_users = events_test[events_test["user_id"].isin(common_users)].copy()
        recs_for_common_users = recs[recs["user_id"].isin(common_users)].copy()

        recs_for_common_users = recs_for_common_users.sort_values(["user_id", "score"], ascending=[True, False])

        # оставляет только те item_id, которые были в events_train, 
        # т. к. модель не имела никакой возможности давать рекомендации для новых айтемов
        events_for_common_users = events_for_common_users[events_for_common_users["item_id"].isin(events_train["item_id"].unique())]

        if top_k is not None:
            recs_for_common_users = recs_for_common_users.groupby("user_id").head(top_k)
        
        events_recs_common = events_for_common_users[["user_id", "item_id", "gt"]].merge(
            recs_for_common_users[["user_id", "item_id", "score"]], 
            on=["user_id", "item_id"], how="outer")    

        events_recs_common["gt"] = events_recs_common["gt"].fillna(False)
        events_recs_common["pr"] = ~events_recs_common["score"].isnull()
        
        events_recs_common["tp"] = events_recs_common["gt"] & events_recs_common["pr"]
        events_recs_common["fp"] = ~events_recs_common["gt"] & events_recs_common["pr"]
        events_recs_common["fn"] = events_recs_common["gt"] & ~events_recs_common["pr"]

        return events_recs_common, recs_for_common_users, events_for_common_users

    def compute_cls_metrics(events_recs_for_binary_metrics):
        groupper = events_recs_for_binary_metrics.groupby("user_id")

        # precision = tp / (tp + fp)
        precision = groupper["tp"].sum()/(groupper["tp"].sum()+groupper["fp"].sum())
        precision = precision.fillna(0).mean()
        
        # recall = tp / (tp + fn)
        recall = groupper["tp"].sum()/(groupper["tp"].sum()+groupper["fn"].sum())
        recall = recall.fillna(0).mean()

        return precision, recall

    # топ популярных
    top_k_pop_items = events_train.groupby('item_id').size().reset_index(name='count').sort_values(['count'], ascending=False)[:10]

    users_train = events_train["user_id"].drop_duplicates()
    users_test = events_test["user_id"].drop_duplicates()

    cold_users = set(users_test) - set(users_train)

    cold_users_events_with_recs = events_test[events_test["user_id"].isin(cold_users)].copy()
    cold_users_events_with_recs["target"] = cold_users_events_with_recs["item_id"].isin(top_k_pop_items["item_id"]).astype(int)

    precision = cold_users_events_with_recs.groupby("user_id")["target"].sum() / 100
    precision_top = precision.mean()

    recall_top = cold_users_events_with_recs.groupby("user_id")["target"].mean().mean()

    cold_user_rankings = {user_id: top_k_pop_items['item_id'].tolist() for user_id in cold_users}
    cold_user_relevant = (events_test[events_test['user_id'].isin(cold_users)]
                            .groupby('user_id')['item_id']
                            .apply(set)
                            .to_dict())

    ndcg_pop = mean_ndcg_at_k(cold_user_rankings, cold_user_relevant, k=10)

    # als metrics
    events_recs_for_binary_metrics, recs_for_common_users, events_for_common_users = process_events_recs_for_binary_metrics(
    events_train,
        events_test, 
        als_recommendations, 
        top_k=10)

    precision_als, recall_als = compute_cls_metrics(events_recs_for_binary_metrics)

    user_rankings = (recs_for_common_users.groupby('user_id')['item_id']
                                            .apply(list)
                                            .to_dict())

    user_relevant = (events_for_common_users.groupby('user_id')['item_id']
                                            .apply(set)
                                            .to_dict())

    ndcg_als = mean_ndcg_at_k(user_rankings, user_relevant, k=10)

    # итоговые

    # для экономии ресурсов оставим события только тех пользователей, 
    # для которых следует оценить рекомендации
    events_inference = pd.concat([events_train, events_test])
    events_inference = events_inference[events_inference["user_id"].isin(events_test["user_id"].drop_duplicates())]

    cb_events_recs_for_binary_metrics_5, recs_for_common_users, events_for_common_users = process_events_recs_for_binary_metrics(
        events_inference,
        events_test,
        final_recommendations.rename(columns={"cb_score": "score"}), 
        top_k=10)

    cb_precision_10, cb_recall_10 = compute_cls_metrics(cb_events_recs_for_binary_metrics_5)

    user_rankings = (recs_for_common_users.groupby('user_id')['item_id']
                                            .apply(list)
                                            .to_dict())

    user_relevant = (events_for_common_users.groupby('user_id')['item_id']
                                            .apply(set)
                                            .to_dict())

    ndcg_cb = mean_ndcg_at_k(user_rankings, user_relevant, k=10)

    # логирование
    os.makedirs('metrics', exist_ok=True)

    metrics = {
            'top10_precision': precision_top,
            'top10_recall': recall_top,
            'top10_NDCG': ndcg_pop,
            'als_precision': precision_als,
            'als_recall': recall_als,
            'cb_precision': cb_precision_10,
            'cb_recall': cb_recall_10,
            'cb_ndcg': ndcg_cb
        }
    with open('metrics/metrics.json', 'w') as fd:
        json.dump(metrics, fd)

    # логирование в mlflow
    load_dotenv()

    TRACKING_SERVER_HOST = "127.0.0.1"
    TRACKING_SERVER_PORT = 5000

    EXPERIMENT_NAME = 'FINAL_PROJECT'

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
    mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

    if mlflow.get_experiment_by_name(name=EXPERIMENT_NAME):
        experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
        experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']
        mlflow.set_experiment(experiment_id=experiment_id)

    pip_requirements = 'requirements.txt'
    cb_model = joblib.load('models/cb_model.pkl')
    params = cb_model.get_params()

    with mlflow.start_run(run_name='cb_model_log', experiment_id=experiment_id) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.catboost.log_model(
            cb_model=cb_model,
            artifact_path='models',
            registered_model_name='cb_ranking_model',
            pip_requirements=pip_requirements,
        )