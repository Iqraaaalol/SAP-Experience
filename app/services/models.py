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


class CrewAlertRequest(BaseModel):
    """Request model for crew alerts."""
    seatNumber: str
    serviceType: str
    message: str
    priority: str = "medium"
