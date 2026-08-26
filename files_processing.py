import pandas as pd

# item_properties
# трансформация даты
item_properties_part1 = pd.read_csv('item_properties_part1.csv')
item_properties_part1['date'] = pd.to_datetime(item_properties_part1['timestamp'], unit='ms')
item_properties_part2 = pd.read_csv('item_properties_part2.csv')
item_properties_part2['date'] = pd.to_datetime(item_properties_part2['timestamp'], unit='ms')

# объединения датафреймов
item_properties = pd.concat([item_properties_part1, item_properties_part2]).drop(columns=['timestamp']).rename(columns={'itemid': 'item_id'})

item_properties.to_parquet('item_properties.parquet', index=False)

# создание отдельного датафрейма с категориями
item_categories = item_properties[item_properties['property'] == 'categoryid'].sort_values('date').drop(columns=['property'])
item_categories = item_categories.groupby('item_id')['value'].last().reset_index(name='category')

item_categories.to_parquet('item_categories.parquet', index=False)

#events
events = pd.read_csv('events.csv')
events['date'] = pd.to_datetime(events['timestamp'], unit='ms')
events = events.drop(columns=['timestamp']).rename(columns={'itemid': 'item_id', 'visitorid': 'user_id'})

events.to_parquet('events.parquet', index=False)

#top100
events_addtocart = events[events['event'] == 'addtocart']
top100 = events_addtocart.groupby('item_id').size().reset_index(name='count').sort_values(['count'], ascending=False)[:100]

top100.to_parquet('top100.parquet', index=False)

# train, test, upd date split
train_time_split_date = pd.to_datetime("2015-08-01")
test_time_split_date = pd.to_datetime("2015-09-01")

train_time_split_date_idx = events["date"] < train_time_split_date
events_train = events[train_time_split_date_idx]

test_time_split_date_idx = (events["date"] >= train_time_split_date) & (events["date"] < test_time_split_date)
events_test = events[test_time_split_date_idx]

upd_time_split_date_idx = events["date"] >= test_time_split_date
events_upd = events[upd_time_split_date_idx]

print(len(events_train), len(events_test), len(events_upd))

events_train.to_parquet('events_train.parquet')
events_test.to_parquet('events_test.parquet')
events_upd.to_parquet('events_upd.parquet')

print('All done')