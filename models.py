"""
Pydantic models for request/response validation
This file is optional - these models can be defined directly in app.py
"""

from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint"""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Natural language question about member data",
        example="When is Layla planning her trip to London?"
    )


class AnswerResponse(BaseModel):
    """Response model for the /ask endpoint"""
    answer: str = Field(
        ..., 
        description="Answer to the question based on member data",
        example="Layla is planning her trip to London in March."
    )