from fastapi import APIRouter
from edgevoice.orchestrator.executor import Executor

# Global Executor instance
executor = Executor()

# Include all sub-routers
from edgevoice.api.routes.text import router as text_router
from edgevoice.api.routes.voice import router as voice_router
from edgevoice.api.routes.permissions import router as permissions_router
from edgevoice.api.routes.skills import router as skills_router
from edgevoice.api.routes.admin import router as admin_router

main_router = APIRouter()
main_router.include_router(text_router)
main_router.include_router(voice_router)
main_router.include_router(permissions_router)
main_router.include_router(skills_router)
main_router.include_router(admin_router)
