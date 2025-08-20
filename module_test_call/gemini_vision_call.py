from google import genai
from google.genai import types
import pathlib,os,json,time
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_from_pdf(pdf_path: str, schema: dict):
    """
    Extracts structured data from a PDF based on a provided schema.
    
    Args:
        pdf_path (str): Path to the PDF file
        schema (dict): JSON schema describing what fields to extract
    
    Returns:
        str: JSON string containing extracted data
    """

    # Upload the PDF
    sample_file = client.files.upload(file=pathlib.Path(pdf_path))

    # Convert schema dict to pretty JSON string for the prompt
    schema_str = json.dumps(schema, indent=2)

    # Dynamic prompt using schema
    prompt = (
        f"Extract data from the PDF tables and return it strictly as JSON of JSON. "
        f"Each extracted object must follow this schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- The key names must exactly match the schema.\n"
        "- Use null for missing values.\n"
        "- Correct text orientation if needed.\n"
        "- Note that: Output only a valid JSON of JSON, no explanations."
    )

    # Generate response with JSON output
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[sample_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return response.text


if __name__ == "__main__":
    pdf_file = "/rahul//Documents/Heffner 7.3.2025 copy.pdf"

    # Example: Schema provided dynamically
    schema = {
        "Policy Number": "string",
        "insured_name": "string",
        "policy_term": "string",
        "total_annual_premium": "number",
        "premium_paid": "number",
        "commission_rate": "string",
        "commission_paid": "number"
    }

    result = extract_from_pdf(pdf_file, schema)
    print(result)
