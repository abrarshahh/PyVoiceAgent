from fastapi import FastAPI
from edgevoice.api.routes import main_router
from edgevoice.core.logging import setup_logger, get_logger

# Setup Logging
setup_logger()
logger = get_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="EdgeVoice Daemon API")
    logger.info("FastAPI app initialized via factory.")

    # Include Routes
    app.include_router(main_router)

    @app.get("/")
    def read_root():
        return {"message": "EdgeVoice Daemon API is running. Use /text-to-text, /text-to-voice or /voice-to-voice."}

    @app.get("/healthz")
    def healthz():
        return {"status": "healthy"}

    return app

app = create_app()
