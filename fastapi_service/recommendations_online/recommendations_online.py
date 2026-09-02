from fastapi import FastAPI
import requests
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

features_store_url = os.environ.get('FEATURES_STORE_URL')
events_store_url = os.environ.get('EVENTS_STORE_URL')

logger = logging.getLogger("uvicorn.error")

# создаём приложение FastAPI
app = FastAPI(title="events")

# prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

events_get_error = Counter('events_get_error_total', 'Count of get events errors')
sim_items_get_error = Counter('sim_items_get_error_total', 'Count of get sim items')


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100):
    """
    Возвращает список онлайн-рекомендаций длиной k для пользователя user_id
    """

    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    # получаем список последних событий пользователя, возьмём три последних
    params = {"user_id": user_id, "k": 3}
    try: 
        response = requests.post(events_store_url + "/get", headers=headers, params=params)
        events = response.json()
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.exception(f'Failed to get events: {e}')
        events_get_error.inc()
        events = []

    # получаем список айтемов, похожих на последние три, с которыми взаимодействовал пользователь
    items = []
    scores = []
    for item_id in events['events']:
        # для каждого item_id получаем список похожих в item_similar_items
        try:
            response = requests.post(features_store_url + "/similar_items", headers=headers, params={'item_id': item_id, 'k': 10})
            item_similar_items = response.json()
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.exception(f'Failed to get sim items: {e}')
            sim_items_get_error.inc()
            item_similar_items = []
        items += item_similar_items["sim_item_id"]
        scores += item_similar_items["score"]
    # сортируем похожие объекты по scores в убывающем порядке
    # для старта это приемлемый подход
    combined = list(zip(items, scores))
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    combined = [item for item, _ in combined]

    # удаляем дубликаты, чтобы не выдавать одинаковые рекомендации
    recs = dedup_ids(combined)

    return {"recs": recs[:k]}