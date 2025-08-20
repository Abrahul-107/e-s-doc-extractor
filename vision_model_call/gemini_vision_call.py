from google import genai
from google.genai import types
import pathlib, os, json
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def clean_newlines(obj):
    """
    Recursively replace \n with space in all string values of a dict/list.
    """
    if isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_newlines(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("\n", "")  # replace newline with space (or "" if you want remove)
    else:
        return obj

def extract_from_pdf(pdf_path: str, schema: dict):
    """
    Extracts structured data from a PDF and returns JSON of JSON (with cleaned strings).
    """

    # Upload the PDF
    sample_file = client.files.upload(file=pathlib.Path(pdf_path))

    # Convert schema dict to pretty JSON string for the prompt
    schema_str = json.dumps(schema, indent=2)

    # Dynamic prompt
    prompt = (
        f"Extract data from the PDF tables and return it strictly as a JSON object of objects. "
        f"Each extracted object must follow this schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- The key names must exactly match the schema.\n"
        "- Use null for missing values.\n"
        "- Correct text orientation if needed.\n"
        "- Output only valid JSON, no explanations."
    )

    # Generate response with JSON output
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[sample_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    # Parse JSON array from Gemini
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON")

    # Convert list → dict of dicts
    result = {}
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            result[str(idx)] = clean_newlines(item) 
    elif isinstance(data, dict):
        result["1"] = clean_newlines(data)

    return result
