from google import genai
from google.genai import types
import pathlib, os, json, logging, time
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

# Initialize Gemini client
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


def call_gemini_with_retry(model: str, contents, max_retries: int = 5, **kwargs):
    """
    Wrapper for Gemini API calls with retry and exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                **kwargs
            )
        except Exception as e:
            error_str = str(e)
            # Retry on known transient errors
            if any(x in error_str for x in ["503", "UNAVAILABLE", "ServiceUnavailable", "Overloaded"]):
                wait_time = min(2 ** attempt, 60)
                logger.warning(f"[Attempt {attempt}] Gemini API overloaded or unavailable. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            logger.error(f"Gemini API call failed permanently: {e}")
            raise HTTPException(status_code=502, detail=f"Gemini API request failed: {e}")
    raise HTTPException(status_code=503, detail="Gemini API repeatedly unavailable after retries.")


def extract_from_pdf(pdf_path: str, schema: dict, total_schema: dict):
    """
    Extract structured data from a PDF and return JSON (with cleaned strings).
    """
    # --- Upload PDF ---
    try:
        logger.info(f"Uploading PDF: {pdf_path}")
        sample_file = client.files.upload(file=pathlib.Path(pdf_path))
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF to Gemini API: {e}")

    schema_str = json.dumps(schema, indent=2)
    total_str = json.dumps(total_schema, indent=2)

    prompt = (
        f"Extract data from the PDF tables and return it strictly as a JSON object of objects. "
        f"Give me all the data, don't miss a single one.\n\n"
        f"Each extracted object must follow this schema:\n{schema_str}\n\n"
        f"At the end, add one JSON object for:\n{total_str}\n\n"
        "Rules:\n"
        "- The key names must exactly match the schema.\n"
        "- Use null for missing values.\n"
        "- Output only valid JSON, no explanations.\n"
        "- Correct text orientation if needed."
    )

    # --- Call Gemini with retries ---
    response = call_gemini_with_retry(
        # model="gemini-2.5-pro",
        model = "gemini-2.5-flash",
        contents=[sample_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    logger.info("Gemini API response received.")

    # --- Validate response ---
    if not response or not getattr(response, "text", None):
        logger.error("Received an empty or invalid response from Gemini API.")
        raise HTTPException(status_code=500, detail="Gemini API returned an empty response.")

    logger.info(f"Received raw text from Gemini: {response.text[:200]}...")  # Log first 200 chars

    # --- Parse and clean JSON ---
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON returned by Gemini: {e}. Attempting repair...")
        try:
            repaired = json_repair.repair_json(response.text)
            data = json.loads(repaired)
            logger.info("Successfully repaired JSON.")
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON.")

    # --- Normalize output ---
    result = {}
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            result[str(idx)] = clean_newlines(item)
    elif isinstance(data, dict):
        result["1"] = clean_newlines(data)
    else:
        logger.error(f"Unexpected data type received from Gemini: {type(data)}")
        raise HTTPException(status_code=500, detail="Gemini returned an unexpected data format.")

    return result
