import pandas as pd

def candidates():
    events_train = pd.read_parquet('data/events_train.parquet')
    events_test = pd.read_parquet('data/events_test.parquet')
    als_recommendations = pd.read_parquet('data/als_recommendations.parquet')

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

    return candidates_for_train, candidates_to_rank