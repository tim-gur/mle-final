import logging
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

logger = logging.getLogger("uvicorn.error")

sim_items_error = Counter('sim_items_error', 'Sim items not found count')

class SimilarItems:

    def __init__(self):

        self._similar_items = None

    def load(self, path, **kwargs):
        """
        Загружаем данные из файла
        """

        logger.info(f"Loading data")
        self._similar_items = pd.read_parquet(path, **kwargs)
        self._similar_items = self._similar_items.set_index("item_id")
        logger.info(f"Loaded")

    def get(self, item_id: int, k: int = 10):
        """
        Возвращает список похожих объектов
        """
        try:
            i2i = self._similar_items.loc[item_id].head(k)
            i2i = i2i[["sim_item_id", "score"]].to_dict(orient="list")
        except KeyError:
            logger.error("No recommendations found")
            i2i = {"sim_item_id": [], "score": []}
            sim_items_error.inc()

        return i2i

sim_items_store = SimilarItems()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # код ниже (до yield) выполнится только один раз при запуске сервиса
    try:
        sim_items_store.load(
            'similar.parquet',
            columns=["item_id", "sim_item_id", "score"],
        )
    except Exception as e:
        logger.exception(f'sim_items_store not loaded: {e}')
        raise

    logger.info("Ready!")
    # код ниже выполнится только один раз при остановке сервиса
    yield

# создаём приложение FastAPI
app = FastAPI(title="features", lifespan=lifespan)

# prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/similar_items")
async def recommendations(item_id: int, k: int = 10):
    """
    Возвращает список похожих объектов длиной k для item_id
    """

    i2i = sim_items_store.get(item_id, k)

    return i2i