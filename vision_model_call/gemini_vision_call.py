from google import genai
from google.genai import types
import pathlib, os, json, logging
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()


# Setup logging ,Load ENV and init Gemini client
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pdf_extractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY in environment variables.")
    raise ValueError("Missing GEMINI_API_KEY in environment variables.")

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



def extract_from_pdf(pdf_path: str, schema: dict):
    """
    Extract structured data from a PDF and return JSON (with cleaned strings).
    """
    try:
        # Upload the PDF
        logger.info(f"Uploading PDF: {pdf_path}")
        sample_file = client.files.upload(file=pathlib.Path(pdf_path))
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF to Gemini API {e}")

    schema_str = json.dumps(schema, indent=2)

    prompt = (
        f"Extract data from the PDF tables and return it strictly as a JSON object of objects. "
        f"Each extracted object must follow this schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- The key names must exactly match the schema.\n"
        "- Use null for missing values.\n"
        "- Correct text orientation if needed.\n"
        "- Output only valid JSON, no explanations."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[sample_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        logger.info("Gemini API response received")
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=502, detail="Gemini API request failed")

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON returned by Gemini: {e}")
        raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

    # Convert list → dict of dicts
    result = {}
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            result[str(idx)] = clean_newlines(item)
    elif isinstance(data, dict):
        result["1"] = clean_newlines(data)

    return result

