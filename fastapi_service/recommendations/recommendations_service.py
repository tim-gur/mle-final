from fastapi import FastAPI
import pandas as pd
import requests
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

recommendations_offline_url = os.environ.get('RECOMMENDATIONS_OFFLINE_URL')
recommendations_online_url = os.environ.get('RECOMMENDATIONS_ONLINE_URL')

logger = logging.getLogger('uvicorn.error')

def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

app = FastAPI(title="recommendations")

# prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

recs_offline_errors = Counter('recs_offline_errors_total', 'Recommendations offline errors')
recs_online_errors = Counter('recs_online_errors_total', 'Recommendations online errors')

@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    params = {"user_id": user_id, 'k': k}

    try:
        resp_offline = requests.post(recommendations_offline_url + "/recommendations_offline", headers=headers, params=params)
        recs_offline = resp_offline.json()["recs"]
        resp_offline.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.exception(f'recs_offline not received: {e}')
        recs_offline_errors.inc()
        recs_offline = []

    try:
        resp_online = requests.post(recommendations_online_url + "/recommendations_online", headers=headers, params=params)
        recs_online = resp_online.json()["recs"]
        resp_online.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.exception(f'recs_online not received: {e}')
        recs_online_errors.inc()
        recs_online = []

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