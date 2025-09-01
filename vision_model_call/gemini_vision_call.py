from google import genai
from google.genai import types
import pathlib, os, json, logging
from dotenv import load_dotenv
import json_repair
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY in environment variables.")
    raise ValueError("Missing GEMINI_API_KEY in environment variables.")

# Initialize Gemini client with the API key
client = genai.Client(api_key=GEMINI_API_KEY)

def clean_newlines(obj):
    """Recursively replace \n with space in all string values."""
    if isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_newlines(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("\n", " ")  
    return obj

def extract_from_pdf(pdf_path: str, schema: dict,total_schema:dict):
    """
    Extract structured data from a PDF and return JSON (with cleaned strings).
    """
    try:
        # Upload the PDF
        logger.info(f"Uploading PDF: {pdf_path}")
        sample_file = client.files.upload(file=pathlib.Path(pdf_path))
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF to Gemini API: {e}")

    schema_str = json.dumps(schema, indent=2)
    total_str = json.dumps(total_schema, indent=2)

    prompt = (
        f"Extract data from the PDF tables and return it strictly as a JSON object of objects. give me all the data don't miss a single one "
        f"Each extracted object must follow this schema:\n{schema_str}\n\n"
        f"- In the end add one json object  for\n{total_str}\n "
        "Rules:\n"
        "- The key names must exactly match the schema.\n"
        "- Use null for missing values.\n"
        "- In the end add one json object  for\n{total_str}\n "
        "- Correct text orientation if needed.\n"
        "- Output only valid JSON, no explanations."
    )

    response = None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[sample_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        logger.info("Gemini API response received")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=502, detail="Gemini API request failed")

    # Handle empty or None response
    if not response or not response.text:
        logger.error("Received an empty or invalid response from Gemini API.")
        raise HTTPException(status_code=500, detail="Gemini API returned an empty response")
    
    # Log the received response for debugging
    logger.info(f"Received raw text from Gemini: {response.text[:200]}...") # Print first 200 chars

    data = None
    try:
        # First, try normal JSON parsing
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON returned by Gemini: {e}. Attempting to repair...")
        try:
            # Attempt to repair JSON if the initial parse fails
            fixed_json = json_repair.repair_json(response.text)
            data = json.loads(fixed_json)
            logger.info("Successfully repaired JSON.")
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")
    
    result = {}
    if isinstance(data, list):
        # Your original code uses a list for multiple items, which is good.
        for idx, item in enumerate(data, start=1):
            result[str(idx)] = clean_newlines(item)
    elif isinstance(data, dict):
        # The prompt asks for an "object of objects", so this handles a single object case.
        result["1"] = clean_newlines(data)
    else:
        # Handle unexpected data types
        logger.error(f"Unexpected data type received from Gemini: {type(data)}")
        raise HTTPException(status_code=500, detail="Gemini returned an unexpected data format")
    
    return result