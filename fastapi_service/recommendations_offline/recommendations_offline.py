import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

logger = logging.getLogger("uvicorn.error")

request_personal_count = Counter('request_personal_count', 'Count of personal requests')
request_default_count = Counter('request_default_count', 'Count of default requests')
recs_error = Counter('recs_not_found', 'Recs not found count')

class Recommendations:

    def __init__(self):

        self._recs = {"personal": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_default_count": 0,
        }

    def load(self, type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """
        logger.info(f"Loading recommendations, type: {type}")
        try:
            self._recs[type] = pd.read_parquet(path, **kwargs)
            if type == "personal":
                self._recs[type] = self._recs[type].set_index("user_id")
            logger.info(f"Loaded")
        except Exception as e:
            logger.exception(f'{type} recommendations file not loaded: {e}')
            raise

    def get(self, user_id: int, k: int=100):
        """
        Возвращает список рекомендаций для пользователя
        """ 
        try:
            recs = self._recs["personal"].loc[user_id]
            recs = recs["item_id"].to_list()[:k]
            request_personal_count.inc()
        except KeyError:
            recs = self._recs["default"]
            recs = recs["item_id"].to_list()[:k]
            request_default_count.inc()
        except Exception as e:
            logger.error(f"No recommendations found: {e}")
            recs_error.inc()
            recs = []
            
        return recs

    def stats(self):
        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ")

rec_store = Recommendations()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # код ниже (до yield) выполнится только один раз при запуске сервиса
    logger.info("Starting")
    rec_store.load(
        "personal",
        'recommendations.parquet',
        columns=["user_id", "item_id", "rank"],
    )
    rec_store.load(
        "default",
        'top100.parquet',
        columns=["item_id", "rank"],
    )
    yield
    # этот код выполнится только один раз при остановке сервиса
    logger.info("Stopping")
    rec_store.stats()
    
# создаём приложение FastAPI
app = FastAPI(title="recommendations_offline", lifespan=lifespan)

# prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """
    recs = rec_store.get(user_id=user_id, k=k)

    return {"recs": recs}