"""Aggregate v1 API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import catalog, health, media, products, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(search.router)
api_router.include_router(products.router)
api_router.include_router(catalog.router)
api_router.include_router(media.router)
