# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import os
from google import genai
from dotenv import load_dotenv
from vision_model_call.gemini_vision_call import extract_from_pdf
# Load ENV
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Init Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

from fastapi import FastAPI

app = FastAPI(
    title="PDF to Structured JSON Extractor",
    description="Extracts structured fields from PDFs using Gemini function calling.",
    version="1.0.0",
    contact={
        "name": "Rahul Parida",
        "url": "https://github.com/Abrahul-107",
        "email": "abrahul882@gmail.com",
    },
    license_info={
        "name": "MIT License",
    },
)

# Request model
class ExtractRequest(BaseModel):
    pdf_path: str
    schema: Dict[str, Any]



@app.post("/extract")
def extract(request: ExtractRequest):
    """
    API endpoint: Extract data from PDF according to schema
    """
    result = extract_from_pdf(request.pdf_path, request.schema)
    return {"data": result}
