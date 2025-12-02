from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from logs.log import logger
import uvicorn


app = FastAPI(
    title="Vendor HelpDesk Agent",
    description="Backend API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "Vendor HelpDesk Agent is okay!",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Starting Vendor HelpDesk Agent API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )