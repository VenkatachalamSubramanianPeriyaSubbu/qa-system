"""
Main FastAPI application for Aurora QA System
Exposes /ask endpoint for natural language question answering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn
from datetime import datetime
import asyncio

from config import get_settings
from data_fetcher import DataFetcher
from qa_processor import QAProcessor
from models import QuestionRequest, AnswerResponse
from data_analyzer import DataAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aurora QA System",
    description="Natural language question-answering system for Aurora member data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
data_fetcher = DataFetcher(request_delay=5.0)
qa_processor = QAProcessor(use_rag=True, top_k=10)
data_analyzer = DataAnalyzer()

member_data_cache = {
    "data": None,
    "timestamp": None,
    "cache_duration": 300
}


async def refresh_data_cache():
    """Refresh the member data cache and index for RAG"""
    try:
        logger.info("Refreshing member data cache...")
        member_data = await data_fetcher.fetch_all_messages()
        member_data_cache["data"] = member_data
        member_data_cache["timestamp"] = datetime.now()
        logger.info(f"Cache refreshed with {len(member_data)} messages")
        
        logger.info("Building knowledge graph...")
        qa_processor.index_messages(member_data)
        logger.info("Knowledge graph built successfully")
        
        return member_data
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise


async def get_cached_data():
    """Get member data from cache or refresh if needed"""
    current_time = datetime.now()
    
    if (member_data_cache["data"] is None or 
        member_data_cache["timestamp"] is None or
        (current_time - member_data_cache["timestamp"]).seconds > member_data_cache["cache_duration"]):
        return await refresh_data_cache()
    
    return member_data_cache["data"]


@app.on_event("startup")
async def startup_event():
    """Initialize cache on startup"""
    logger.info("Starting Aurora QA System...")
    try:
        await refresh_data_cache()
        logger.info("Initial data cache loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load initial cache: {e}")


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "running",
        "service": "Aurora QA System",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Submit a natural language question",
            "/health": "GET - Health check",
            "/analyze": "GET - Data analysis insights"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    cache_status = "loaded" if member_data_cache["data"] else "empty"
    return {
        "status": "healthy",
        "cache_status": cache_status,
        "cache_timestamp": member_data_cache["timestamp"].isoformat() if member_data_cache["timestamp"] else None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for answering natural language questions about member data
    
    Args:
        request: QuestionRequest with the question field
    
    Returns:
        AnswerResponse with the answer field
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        member_data = await get_cached_data()
        
        if not member_data:
            raise HTTPException(
                status_code=503, 
                detail="Unable to fetch member data. Please try again later."
            )
        
        answer = await qa_processor.process_question(
            question=request.question,
            member_data=member_data
        )
        
        logger.info(f"Generated answer: {answer}")
        
        return AnswerResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/analyze")
async def analyze_data():
    """
    Analyze dataset for anomalies and insights
    """
    try:
        logger.info("Running data analysis...")
        
        member_data = await get_cached_data()
        
        if not member_data:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch member data for analysis."
            )
        
        insights = await data_analyzer.analyze(member_data)
        
        return {
            "status": "success",
            "insights": insights,
            "total_messages": len(member_data),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing data: {str(e)}"
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=settings.DEBUG
    )