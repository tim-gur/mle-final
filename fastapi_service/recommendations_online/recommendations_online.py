from fastapi import FastAPI
import requests

features_store_url = "http://features_service:8010"
events_store_url = "http://events_service:8020"

# создаём приложение FastAPI
app = FastAPI(title="events")

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
    events = requests.post(events_store_url + "/get", headers=headers, params=params).json()

    # получаем список айтемов, похожих на последние три, с которыми взаимодействовал пользователь
    items = []
    scores = []
    for item_id in events['events']:
        # для каждого item_id получаем список похожих в item_similar_items
        item_similar_items = requests.post(features_store_url + "/similar_items", headers=headers, params={'item_id': item_id, 'k': 10}).json()
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