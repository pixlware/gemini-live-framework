"""
API Routes for Classplus Calling Bot.
Handles incoming HTTP requests, Exotel webhooks, and WebSocket streaming.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "status": "healthy",
        "version": "1.0.0"
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "service": settings.APP_NAME,
        "status": "healthy",
        "version": "1.0.0"
    }

# --- WebSocket Endpoints ---------------------------------------------

@router.websocket("/ws/media-stream")
async def websocket_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Gemini Live Framework.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        pass
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")
