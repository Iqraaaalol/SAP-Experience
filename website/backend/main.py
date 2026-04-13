from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import uvicorn

app = FastAPI(
    title="Vector4Solutions API",
    description="Backend API for Vector4Solutions - Aviation Intelligence Platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models

class ContactRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    message: str = Field(..., min_length=10, max_length=2000)
    service_type: Optional[str] = Field(None, max_length=50)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "message": "I'm interested in learning more about your AI-powered passenger intelligence systems.",
                "service_type": "Passenger Experience"
            }
        }


class ContactResponse(BaseModel):
    success: bool
    message: str


# API Endpoints

@app.get("/")
async def root():
    return {"message": "Vector4Solutions API is Online"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Vector4Solutions API", "version": "1.0.0"}


@app.post("/api/contact", response_model=ContactResponse)
async def contact(contact_data: ContactRequest):
    print("\n" + "=" * 60)
    print("NEW CONTACT FORM SUBMISSION")
    print("=" * 60)
    print(f"Name: {contact_data.name}")
    print(f"Email: {contact_data.email}")
    print(f"Service Type: {contact_data.service_type or 'Not specified'}")
    print(f"Message: {contact_data.message}")
    print("=" * 60 + "\n")

    return ContactResponse(
        success=True,
        message="Thank you for contacting Vector4Solutions. We will get back to you soon."
    )


# Run the Application

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
