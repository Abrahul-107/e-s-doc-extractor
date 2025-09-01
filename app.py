# app.py
from fastapi import FastAPI
from vision_model_call.gemini_vision_call import extract_from_pdf
from vision_model_call.call_for_header import extract_table_headers
from fastapi import FastAPI
from model.models import ExtractRequest,ExtractHeader
import logging,uvicorn,os
from google import genai
from fastapi.responses import JSONResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")



# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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



@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns the status of the application, environment variables, and Gemini API connectivity.
    """
    status = {"app": "ok", "environment": {}, "gemini": {}}

    # Check environment variables
    status["environment"]["GEMINI_API_KEY"] = "set" if GEMINI_API_KEY else "missing"

    # Check Gemini API
    if gemini_client:
        try:
            # Quick test request: list available models (or another lightweight call)
            models = gemini_client.models.list()
            status["gemini"]["status"] = "reachable"
            status["gemini"]["available_models"] = len(models)
        except Exception as e:
            status["gemini"]["status"] = "unreachable"
            status["gemini"]["error"] = str(e)
    else:
        status["gemini"]["status"] = "not initialized"

    return JSONResponse(content=status, status_code=200)


@app.post("/extract")
def extract(request: ExtractRequest):
    """
    Extract data from PDF according to schema.
    """
    logger.info(f"Received extract request for {request.pdf_path}")
    result = extract_from_pdf(request.pdf_path, request.schema,request.total_schema)
    return {"data": result}


@app.post("/get_header")
def get_headers(request: ExtractHeader):
    """
    Extract only table headers from PDF and map to schema keys.
    """
    if not gemini_client:
        return JSONResponse(content={"error": "Gemini client not initialized"}, status_code=500)

    logger.info(f"Received header extraction request for {request.pdf_path}")
    result = extract_table_headers(request.pdf_path, request.schema)
    return {"headers": result}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",  
        access_log=False      
    )