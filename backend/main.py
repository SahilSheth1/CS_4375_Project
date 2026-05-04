from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import model_loader
from routers import upload, review


@asynccontextmanager
async def lifespan(app: FastAPI):
    (
        model_loader.model,
        model_loader.vocab,
        model_loader.temperatures,
        model_loader.thresholds,
    ) = model_loader.load_model()

    # add src directory to sys.path to import local modules
    import sys
    from pathlib import Path

    # resolves the absolute path to the src directory
    sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
    from conf_scoring_class import ConfidenceScorer


    model_loader.scorer = ConfidenceScorer(
        model=model_loader.model,
        vocab=model_loader.vocab,
        device=model_loader.device,
    )
    print("[startup] ConfidenceScorer ready")
    yield
    print("[shutdown] Cleaning up")

# initialize FastAPI with metadata for Swagger
app = FastAPI(
    title="Receipt Digitization API",
    description="Transformer-based Vision Encoder for Automated Receipt Digitization",
    version="1.0.0",
    lifespan=lifespan,
)

# configure CORS and allow React to communicate with backed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register endpoint routers for code structure
app.include_router(upload.router)
app.include_router(review.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Receipt Digitization API is running"}
