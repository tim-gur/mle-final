import pendulum
from airflow.decorators import dag, task

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2015, 5, 3, tz="UTC"),
    catchup=False,
    tags=["ETL"]
)
def pipeline_retrain():
    import pandas as pd
    import os
    from sklearn.preprocessing import LabelEncoder
    import joblib
    import scipy
    import numpy as np
    from implicit.als import AlternatingLeastSquares
    from catboost import CatBoostClassifier, Pool
    import json
    import mlflow

    DATA_DIR = os.environ.get("DATA_DIR")
    DOWNLOADS_DIR = os.environ.get("DOWNLOADS_DIR")
    MODELS_DIR = os.environ.get("MODELS_DIR")
    METRICS_DIR = os.environ.get("METRICS_DIR")

    @task()
    def process_data_retrain():
        # item_properties
        # трансформация даты
        item_properties_part1 = pd.read_csv(os.path.join(DOWNLOADS_DIR, 'item_properties_part1.csv'))
        item_properties_part1['date'] = pd.to_datetime(item_properties_part1['timestamp'], unit='ms')
        item_properties_part2 = pd.read_csv(os.path.join(DOWNLOADS_DIR, 'item_properties_part2.csv'))
        item_properties_part2['date'] = pd.to_datetime(item_properties_part2['timestamp'], unit='ms')

        # объединения датафреймов
        item_properties = pd.concat([item_properties_part1, item_properties_part2]).drop(columns=['timestamp']).rename(columns={'itemid': 'item_id'})

        item_properties.to_parquet(os.path.join(DATA_DIR, 'item_properties.parquet'), index=False)

        # создание отдельного датафрейма с категориями
        item_categories = item_properties[item_properties['property'] == 'categoryid'].sort_values('date').drop(columns=['property'])
        item_categories = item_categories.groupby('item_id')['value'].last().reset_index(name='category')

        item_categories.to_parquet(os.path.join(DATA_DIR, 'item_categories.parquet'), index=False)

        #category_tree
        category_tree = pd.read_csv(os.path.join(DOWNLOADS_DIR, 'category_tree.csv'))
        category_tree = category_tree.dropna()
        category_tree['parentid'] = category_tree['parentid'].astype('int32')

        category_tree.to_csv(os.path.join(DATA_DIR, 'category_tree.csv'))

        #events
        events = pd.read_csv(os.path.join(DOWNLOADS_DIR, 'events.csv'))
        events['date'] = pd.to_datetime(events['timestamp'], unit='ms')
        events = events.drop(columns=['timestamp']).rename(columns={'itemid': 'item_id', 'visitorid': 'user_id'})

        events.to_parquet(os.path.join(DATA_DIR, 'events.parquet'), index=False)

        #top100
        events_addtocart = events[events['event'] == 'addtocart']
        top100 = events_addtocart.groupby('item_id').size().reset_index(name='count').sort_values(['count'], ascending=False)[:100]
        top100['rank'] = top100['count'].rank(method='first', ascending=False).astype(int)

        top100.to_parquet(os.path.join(DATA_DIR, 'top100.parquet'), index=False)

        # train, test, upd date split
        train_time_split_date = pd.to_datetime("2015-08-01")
        test_time_split_date = pd.to_datetime("2015-09-01")

        train_time_split_date_idx = events["date"] < train_time_split_date
        events_train = events[train_time_split_date_idx]

        test_time_split_date_idx = (events["date"] >= train_time_split_date) & (events["date"] < test_time_split_date)
        events_test = events[test_time_split_date_idx]

        upd_time_split_date_idx = events["date"] >= test_time_split_date
        events_upd = events[upd_time_split_date_idx]

        events_train = pd.concat([events_train, events_upd])
        events_train.to_parquet(os.path.join(DATA_DIR, 'events_train.parquet'))

        events_test.to_parquet(os.path.join(DATA_DIR, 'events_test.parquet'))
        

    @task()
    def als_recommendations():
        events_train = pd.read_parquet(os.path.join(DATA_DIR, 'events_train.parquet'))
        events_test = pd.read_parquet(os.path.join(DATA_DIR, 'events_test.parquet'))
        events = pd.read_parquet(os.path.join(DATA_DIR, 'events.parquet'))

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
        joblib.dump(als_model, os.path.join(MODELS_DIR, 'als_model.pkl'))

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

        als_recommendations.to_parquet(os.path.join(DATA_DIR, 'als_recommendations.parquet'))

        # построение тиблицы похожих предметов
        # получаем список всех возможных item_id (перекодированных)
        item_ids_encoded = range(len(item_encoder.classes_))

        similar = als_model.similar_items(item_ids_encoded, N=5)

        # преобразуем полученные рекомендации в табличный формат
        item_ids_enc = similar[0]
        als_scores = similar[1]

        similar_recommendations = pd.DataFrame({
            "item_id_enc": item_ids_encoded,
            "sim_item_id_enc": item_ids_enc.tolist(), 
            "score": als_scores.tolist()})
        similar_recommendations = similar_recommendations.explode(["sim_item_id_enc", "score"], ignore_index=True)

        # приводим типы данных
        similar_recommendations["sim_item_id_enc"] = similar_recommendations["sim_item_id_enc"].astype("int")
        similar_recommendations["score"] = similar_recommendations["score"].astype("float")

        # получаем изначальные идентификаторы
        similar_recommendations["item_id"] = item_encoder.inverse_transform(similar_recommendations["item_id_enc"])
        similar_recommendations["sim_item_id"] = item_encoder.inverse_transform(similar_recommendations["sim_item_id_enc"])
        similar_recommendations = similar_recommendations.drop(columns=["item_id_enc", 'sim_item_id_enc'])

        similar_recommendations.to_parquet(os.path.join(DATA_DIR, 'similar.parquet'))

    @task()
    def candidates():
        events_train = pd.read_parquet(os.path.join(DATA_DIR, 'events_train.parquet'))
        events_test = pd.read_parquet(os.path.join(DATA_DIR, 'events_test.parquet'))
        als_recommendations = pd.read_parquet(os.path.join(DATA_DIR, 'als_recommendations.parquet'))

        # добавляем таргет к кандидатам со значением:
        # — 1 для событий addtocart
        # — 0, для всех остальных 

        events_train_addtocart = events_train[events_train['event'] == 'addtocart'][['user_id', 'item_id']].copy()
        events_train_addtocart['target'] = 1
        candidates = als_recommendations.merge(events_train_addtocart[["user_id", "item_id", "target"]], 
                                    on=['user_id', 'item_id'],
                                    how='left')
        candidates["target"] = candidates["target"].fillna(0).astype("int")

        # в кандидатах оставляем только тех пользователей, у которых есть хотя бы один положительный таргет
        candidates_to_sample = candidates.groupby("user_id").filter(lambda x: x["target"].sum() > 0)

        # для каждого пользователя оставляем только 4 негативных примера
        negatives_per_user = 4
        candidates_for_train = pd.concat([
            candidates_to_sample[candidates_to_sample['target'] == 1],
            candidates_to_sample.query("target == 0") \
                .groupby("user_id") \
                .apply(lambda x: x.sample(min(len(x), negatives_per_user), random_state=42))
            ])

        candidates_to_rank = als_recommendations[als_recommendations["user_id"].isin(events_test["user_id"].drop_duplicates())]\

        candidates_for_train.to_parquet(os.path.join(DATA_DIR, 'candidates_for_train.parquet'), index=False)
        candidates_to_rank.to_parquet(os.path.join(DATA_DIR, 'candidates_to_rank.parquet'), index=False)

    @task()
    def features():
        items = pd.read_parquet(os.path.join(DATA_DIR, 'item_categories.parquet'))
        events_train = pd.read_parquet(os.path.join(DATA_DIR, 'events_train.parquet'))
        events_test = pd.read_parquet(os.path.join(DATA_DIR, 'events_test.parquet'))
        candidates_for_train = pd.read_parquet(os.path.join(DATA_DIR, 'candidates_for_train.parquet'))
        candidates_to_rank = pd.read_parquet(os.path.join(DATA_DIR, 'candidates_to_rank.parquet'))
        category_tree = pd.read_csv(os.path.join(DOWNLOADS_DIR, 'category_tree.csv'))

        # общая категория
        category_tree_dict = dict(zip(category_tree['categoryid'], category_tree['parentid']))

        def get_root_category(cat_id, tree_dict):
            while cat_id in tree_dict:
                cat_id = tree_dict[cat_id]
            return int(cat_id)

        items['general_category'] = items['category'].astype('int64').apply(lambda c: get_root_category(c, category_tree_dict))

        # просмотры пары юзер-айтем
        user_item_views_train = events_train[events_train['event'] == 'view'].groupby(['user_id', 'item_id']).size().reset_index(name='user_item_views_count')
        user_item_views_test = events_test[events_test['event'] == 'view'].groupby(['user_id', 'item_id']).size().reset_index(name='user_item_views_count')

        # просмотры
        user_total_views_train = events_train[events_train['event'] == 'view'].groupby('user_id').size().reset_index(name='user_views_count')
        user_total_views_test = events_test[events_test['event'] == 'view'].groupby('user_id').size().reset_index(name='user_views_count')

        item_total_views_train = events_train[events_train['event'] == 'view'].groupby('item_id').size().reset_index(name='item_views_count')
        item_total_views_test = events_test[events_test['event'] == 'view'].groupby('item_id').size().reset_index(name='item_views_count')

        # добавления в корзину
        user_total_addtocart_train = events_train[events_train['event'] == 'addtocart'].groupby('user_id').size().reset_index(name='user_addtocart_count')
        user_total_addtocart_test = events_test[events_test['event'] == 'addtocart'].groupby('user_id').size().reset_index(name='user_addtocart_count')

        item_total_addtocart_train = events_train[events_train['event'] == 'addtocart'].groupby('item_id').size().reset_index(name='item_addtocart_count')
        item_total_addtocart_test = events_test[events_test['event'] == 'addtocart'].groupby('item_id').size().reset_index(name='item_addtocart_count')

        # все события
        user_total_events_train = events_train.groupby('user_id').size().reset_index(name='user_events_count')
        user_total_events_test = events_test.groupby('user_id').size().reset_index(name='user_events_count')

        item_total_events_train = events_train.groupby('item_id').size().reset_index(name='item_events_count')
        item_total_events_test = events_test.groupby('item_id').size().reset_index(name='item_events_count')

        # совмещение всех признаков для обучения
        candidates_for_train = (
            candidates_for_train
            .merge(user_item_views_train, on=['user_id', 'item_id'], how='left')
            .merge(user_total_views_train, on='user_id', how='left')
            .merge(user_total_addtocart_train, on='user_id', how='left')
            .merge(item_total_views_train, on='item_id', how='left')
            .merge(item_total_addtocart_train, on='item_id', how='left')
            .merge(user_total_events_train, on='user_id', how='left')
            .merge(item_total_events_train, on='item_id', how='left')
            .merge(items, on='item_id', how='left')
        )

        candidates_to_rank = (
            candidates_to_rank
            .merge(user_item_views_test, on=['user_id', 'item_id'], how='left')
            .merge(user_total_views_test, on='user_id', how='left')
            .merge(user_total_addtocart_test, on='user_id', how='left')
            .merge(item_total_views_test, on='item_id', how='left')
            .merge(item_total_addtocart_test, on='item_id', how='left')
            .merge(user_total_events_test, on='user_id', how='left')
            .merge(item_total_events_test, on='item_id', how='left')
            .merge(items, on='item_id', how='left')
        )

        # предобработка данных перед обучением
        cols_to_fix = ['user_item_views_count', 'user_views_count', 'item_views_count', 'user_addtocart_count', 
                    'item_addtocart_count', 'user_events_count', 'item_events_count', 'general_category', 'category']

        candidates_for_train[cols_to_fix] = candidates_for_train[cols_to_fix].fillna(0).astype('int32')
        candidates_for_train['target'] = candidates_for_train['target'].astype('int8')

        candidates_for_train = candidates_for_train.fillna(0)
        candidates_to_rank = candidates_to_rank.fillna(0)

        candidates_to_rank[cols_to_fix] = candidates_to_rank[cols_to_fix].astype('int32')

        candidates_for_train = candidates_for_train.rename(columns={'score': 'als_score'})
        candidates_to_rank = candidates_to_rank.rename(columns={'score': 'als_score'})

        candidates_for_train.to_parquet(os.path.join(DATA_DIR, 'candidates_for_train.parquet'))
        candidates_to_rank.to_parquet(os.path.join(DATA_DIR, 'candidates_to_rank.parquet'))

    @task()
    def fit():
        candidates_for_train = pd.read_parquet(os.path.join(DATA_DIR, 'candidates_for_train.parquet'))
        candidates_to_rank = pd.read_parquet(os.path.join(DATA_DIR, 'candidates_to_rank.parquet'))

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
            random_seed=42,
        )

        # тренируем модель
        cb_model.fit(train_data)

        joblib.dump(cb_model, os.path.join(MODELS_DIR, 'cb_model.pkl'))

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

        final_recommendations.to_parquet(os.path.join(DATA_DIR, 'recommendations.parquet'))

    @task()
    def evaluate():
        events_train = pd.read_parquet(os.path.join(DATA_DIR, 'events_train.parquet'))
        events_test = pd.read_parquet(os.path.join(DATA_DIR, 'events_test.parquet'))
        final_recommendations = pd.read_parquet(os.path.join(DATA_DIR, 'recommendations.parquet'))
        als_recommendations = pd.read_parquet(os.path.join(DATA_DIR, 'als_recommendations.parquet'))

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
        top_k_pop_items = events_train[events_train['event'] == 'addtocart'].groupby('item_id').size().reset_index(name='count').sort_values(['count'], ascending=False)[:10]

        users_train = events_train["user_id"].drop_duplicates()
        users_test = events_test["user_id"].drop_duplicates()

        cold_users = set(users_test) - set(users_train)

        events_test_addtocart = events_test[events_test['event'] == 'addtocart']
        cold_users_events_with_recs = events_test_addtocart[events_test["user_id"].isin(cold_users)].copy()
        cold_users_events_with_recs["target"] = cold_users_events_with_recs["item_id"].isin(top_k_pop_items["item_id"]).astype(int)

        precision = cold_users_events_with_recs.groupby("user_id")["target"].sum() / 100
        precision_top = precision.mean()

        recall_top = cold_users_events_with_recs.groupby("user_id")["target"].mean().mean()

        cold_user_rankings = {user_id: top_k_pop_items['item_id'].tolist() for user_id in cold_users}
        cold_user_relevant = (events_test_addtocart[events_test_addtocart['user_id'].isin(cold_users)]
                            .groupby('user_id')['item_id']
                            .apply(set)
                            .to_dict())

        ndcg_pop = mean_ndcg_at_k(cold_user_rankings, cold_user_relevant, k=10)

        # als metrics
        events_recs_for_binary_metrics, recs_for_common_users, events_for_common_users = process_events_recs_for_binary_metrics(
        events_train,
            events_test[events_test['event'] == 'addtocart'], 
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
            events_test[events_test['event'] == 'addtocart'],
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

        # novelty
        # разметим каждую рекомендацию признаком played
        addtocart_train = events_train[events_train['event'] == 'addtocart'][['user_id', 'item_id']].drop_duplicates()
        addtocart_train["played"] = True
        final_recommendations = final_recommendations.merge(addtocart_train, on=["user_id", "item_id"], how="left")
        final_recommendations["played"] = final_recommendations["played"].fillna(False).astype("bool")

        # проставим ранги
        final_recommendations = final_recommendations.sort_values(by='cb_score', ascending=False)
        final_recommendations["rank"] = final_recommendations.groupby("user_id").cumcount() + 1

        # посчитаем novelty по пользователям
        novelty_5_cb = (1-final_recommendations.query("rank <= 5").groupby("user_id")["played"].mean())

        # coverage
        items = pd.read_parquet(os.path.join(DATA_DIR, 'item_categories.parquet'))

        n_init = items['item_id'].nunique()
        n_als = final_recommendations['item_id'].nunique()

        cov_items_cb = n_als / n_init

        # логирование
        metrics = {
                'top10_precision': precision_top,
                'top10_recall': recall_top,
                'top10_NDCG': ndcg_pop,
                'als_precision': precision_als,
                'als_recall': recall_als,
                'als_ndcg': ndcg_als,
                'cb_precision': cb_precision_10,
                'cb_recall': cb_recall_10,
                'cb_ndcg': ndcg_cb,
                'cb_novelty': novelty_5_cb.mean(),
                'cb_coverage': cov_items_cb
            }
        with open(os.path.join(METRICS_DIR, 'metrics.json'), 'w') as fd:
            json.dump(metrics, fd)

        # логирование в mlflow
        TRACKING_SERVER_HOST = "host.docker.internal"
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

        pip_requirements = (os.path.join(DATA_DIR, 'requirements.txt'))
        cb_model = joblib.load(os.path.join(MODELS_DIR, 'cb_model.pkl'))
        params = cb_model.get_params()

        importances = cb_model.get_feature_importance(prettified=True)
        importances.to_csv(os.path.join(METRICS_DIR, "feature_importance.csv"), index=False)

        with mlflow.start_run(run_name='cb_model_log_airflow', experiment_id=experiment_id) as run:
            mlflow.log_metrics(metrics)
            mlflow.log_params(params)
            mlflow.log_artifact(os.path.join(METRICS_DIR, "feature_importance.csv"))
            mlflow.catboost.log_model(
                cb_model=cb_model,
                artifact_path='models',
                registered_model_name='cb_ranking_model',
                pip_requirements=pip_requirements,
            )

    t1 = process_data_retrain()
    t2 = als_recommendations()
    t3 = candidates()
    t4 = features()
    t5 = fit()
    t6 = evaluate()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6

pipeline_retrain()