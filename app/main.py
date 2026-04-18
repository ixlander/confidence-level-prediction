from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.ml.service import model_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)
