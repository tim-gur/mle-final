import pandas as pd

def process_data():
    # item_properties
    # трансформация даты
    item_properties_part1 = pd.read_csv('downloads/item_properties_part1.csv')
    item_properties_part1['date'] = pd.to_datetime(item_properties_part1['timestamp'], unit='ms')
    item_properties_part2 = pd.read_csv('downloads/item_properties_part2.csv')
    item_properties_part2['date'] = pd.to_datetime(item_properties_part2['timestamp'], unit='ms')

    # объединения датафреймов
    item_properties = pd.concat([item_properties_part1, item_properties_part2]).drop(columns=['timestamp']).rename(columns={'itemid': 'item_id'})

    item_properties.to_parquet('data/item_properties.parquet', index=False)

    # создание отдельного датафрейма с категориями
    item_categories = item_properties[item_properties['property'] == 'categoryid'].sort_values('date').drop(columns=['property'])
    item_categories = item_categories.groupby('item_id')['value'].last().reset_index(name='category')

    item_categories.to_parquet('data/item_categories.parquet', index=False)

    #category_tree
    category_tree = pd.read_csv('downloads/category_tree.csv')
    category_tree = category_tree.dropna()
    category_tree['parentid'] = category_tree['parentid'].astype('int32')

    category_tree.to_csv('data/category_tree.csv')

    #events
    events = pd.read_csv('downloads/events.csv')
    events['date'] = pd.to_datetime(events['timestamp'], unit='ms')
    events = events.drop(columns=['timestamp']).rename(columns={'itemid': 'item_id', 'visitorid': 'user_id'})

    events.to_parquet('data/events.parquet', index=False)

    #top100
    events_addtocart = events[events['event'] == 'addtocart']
    top100 = events_addtocart.groupby('item_id').size().reset_index(name='count').sort_values(['count'], ascending=False)[:100]
    top100['rank'] = top100['count'].rank(method='first', ascending=False).astype(int)

    top100.to_parquet('data/top100.parquet', index=False)

    # train, test, upd date split
    train_time_split_date = pd.to_datetime("2015-08-01")
    test_time_split_date = pd.to_datetime("2015-09-01")

    train_time_split_date_idx = events["date"] < train_time_split_date
    events_train = events[train_time_split_date_idx]

    test_time_split_date_idx = (events["date"] >= train_time_split_date) & (events["date"] < test_time_split_date)
    events_test = events[test_time_split_date_idx]

    upd_time_split_date_idx = events["date"] >= test_time_split_date
    events_upd = events[upd_time_split_date_idx]

    events_train.to_parquet('data/events_train.parquet')
    events_test.to_parquet('data/events_test.parquet')
    events_upd.to_parquet('data/events_upd.parquet')

if __name__ == '__main__':
    process_data() 