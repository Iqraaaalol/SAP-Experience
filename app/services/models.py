"""
Pydantic models for request/response schemas.
"""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for passenger travel queries."""
    query: str
    destination: str
    seatNumber: str
    language: str = "en"


class FeedbackRequest(BaseModel):
    """Request model for comfort feedback."""
    seatNumber: str
    temperature: float
    lighting: float
    noise_level: float
    overall_comfort: str


class CrewAlertRequest(BaseModel):
    """Request model for crew alerts."""
    seatNumber: str
    serviceType: str
    message: str
    priority: str = "medium"
