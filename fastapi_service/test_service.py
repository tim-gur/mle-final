import requests

#uvicorn recommendations_offline:app --port 8000
#uvicorn features_service:app --port 8010 
#uvicorn events_service:app --port 8020 
#uvicorn recommendations_online:app --port 8030 
#uvicorn recommendations:app --port 8040 

recommendations_offline_url = "http://127.0.0.1:8000"
features_store_url = "http://127.0.0.1:8010"
events_store_url = "http://127.0.0.1:8020"
recommendations_online_url = "http://127.0.0.1:8030"
recommendations_url = "http://127.0.0.1:8040"


headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

# для пользователя без персональных рекомендаций
params = {"user_id": 123900, 'k': 10}
resp_top = requests.post(recommendations_offline_url + "/recommendations_offline", headers=headers, params=params)

# для пользователя с персональными рекомендациями, но без онлайн-истории
params = {"user_id": 1115064, 'k': 10}
resp_offline = requests.post(recommendations_offline_url + "/recommendations_offline", headers=headers, params=params)

# для пользователя с персональными рекомендациями и онлайн-историей
user_id = 1228296
event_item_ids =  [38646012, 60292250, 41028870, 69542]
for event_item_id in event_item_ids:
    resp = requests.post(events_store_url + "/put", headers=headers, params={"user_id": user_id, "item_id": event_item_id})

params = {"user_id": 1374582, 'k': 10}
resp_blended = requests.post(recommendations_url + "/recommendations", headers=headers, params=params)

resp_top = resp_top.json()["recs"]
resp_offline = resp_offline.json()["recs"]
recs_blended = resp_blended.json()["recs"]

with open('test_service.log', 'a') as f:
    f.write(f'No personal recs user: {resp_top}\n')
    f.write(f'Personal recs user, no event history: {resp_offline}\n')
    f.write(f'Personal recs user with event history: {recs_blended}\n')

print(f'No personal recs user: {resp_top}')
print(f'Personal recs user, no event history: {resp_offline}')
print(f'Personal recs user with event history: {recs_blended}')

