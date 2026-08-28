import pandas as pd

def features(candidates_for_train, candidates_to_rank):
    items = pd.read_parquet('data/item_categories.parquet')
    events_train = pd.read_parquet('data/events_train.parquet')
    events_test = pd.read_parquet('data/events_test.parquet')
    category_tree = pd.read_csv('downloads/category_tree.csv')

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

    candidates_for_train.to_parquet('data/candidates_for_train.parquet')
    candidates_to_rank.to_parquet('data/candidates_to_rank.parquet')