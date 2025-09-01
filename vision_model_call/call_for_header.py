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


def extract_table_headers(pdf_path: str, schema: dict):
    """
    Extract table headers from a PDF and map them to the input schema keys.
    Returns a JSON object:
      {
          "mapped_schema": {schema_key: matched_header or key},
          "headers_only": [list of all headers]
      }
    """
    try:
        logger.info(f"Uploading PDF: {pdf_path}")
        sample_file = client.files.upload(file=pathlib.Path(pdf_path))
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {e}")

    prompt = (
        f"Extract all table headers from the PDF and map them to the following database schema keys:\n{schema}\n\n"
        "Instructions:\n"
        "- Analyze first 10 row fist"
        "- The schema keys are database column names. Map each key to the best similiar header and the header should be in that type only if it is date it should be give the date header only so on.\n"
        "- Use semantic intelligence and it should be a single header if they refer to the same data.\n"
        "- Normalize headers: if a header contains extra descriptors or abbreviations, reduce it to its core meaning.Example: Effective PPE → Effective"
        "- Normalize headers: remove duplicates, extra spaces, and punctuation.\n"
        "- If no header found in pdf as per schema key, use NULL.\n"
        "- Return two JSON objects only:\n"
        "    1. 'mapped_schema': keys are schema keys,  value should be similiar as the document headers And Not found if not found in the doc.\n"
        "    2. 'headers_only': list of all distinct headers, exactly as they appear or normalized.\n"
        "- Output valid JSON only, no explanations."
    )



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

    if not response or not response.text:
        logger.error("Received empty response from Gemini.")
        raise HTTPException(status_code=500, detail="Empty response from Gemini API")

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON returned. Attempting to repair...")
        try:
            fixed_json = json_repair.repair_json(response.text)
            data = json.loads(fixed_json)
        except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            raise HTTPException(status_code=500, detail="Gemini returned invalid JSON")

    # If Gemini returned a list, pick the first dict that has mapped_schema
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "mapped_schema" in item:
                data = item
                break
        else:
            # fallback if nothing found
            data = {"mapped_schema": {}, "headers_only": []}
    elif not isinstance(data, dict):
        # fallback for unexpected types
        data = {"mapped_schema": {}, "headers_only": []}

    # Clean up string values
    data = clean_newlines(data)

    # Ensure all schema keys exist
    mapped_schema = data.get("mapped_schema", {})
    for key in schema.keys():
        if key not in mapped_schema or not mapped_schema[key]:
            mapped_schema[key] = key  # fallback to schema key itself

    headers_only = data.get("headers_only", [])

    return {
        "mapped_schema": mapped_schema,
        "headers_only": headers_only
    }
