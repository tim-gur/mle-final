from fastapi import FastAPI
import pandas as pd
import requests

def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

recommendations_offline_url = 'http://recommendations_offline:8000'
recommendations_online_url = 'http://recommendations_online:8030'

app = FastAPI(title="recommendations")
@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    params = {"user_id": user_id, 'k': k}

    resp_offline = requests.post(recommendations_offline_url + "/recommendations_offline", headers=headers, params=params)
    recs_offline = resp_offline.json()["recs"]

    resp_online = requests.post(recommendations_online_url + "/recommendations_online", headers=headers, params=params)
    recs_online = resp_online.json()["recs"]

    recs_blended = []

    min_length = min(len(recs_offline), len(recs_online))
    # чередуем элементы из списков, пока позволяет минимальная длина
    for i in range(min_length):
        recs_blended.append(recs_online[i])
        recs_blended.append(recs_offline[i])

    # добавляем оставшиеся элементы в конец
    recs_blended += recs_offline[min_length:]
    recs_blended += recs_online[min_length:]

    # удаляем дубликаты
    recs_blended = dedup_ids(recs_blended)
    
    # оставляем только первые k рекомендаций
    recs_blended = recs_blended[:k]

    return {"recs": recs_blended}