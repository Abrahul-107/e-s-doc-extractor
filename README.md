# e-s-doc-extractor
AI-powered text extraction tool for parsing and analyzing documents.

# 📑 PDF to Structured JSON Extractor (FastAPI + Gemini Function Calling)

This project extracts structured fields (like **Policy Number, Insured Name, Premiums, Commission, etc.**) from PDF insurance documents using **Google Gemini function calling**.  

It uses **FastAPI** for serving the API and supports **Swagger UI** for testing.

---

## 🚀 Features
- Upload **PDF path** and a **JSON schema of fields** you want to extract.
- Dynamically generates a prompt for Gemini.
- Cleans extracted values (e.g., removes `\n` from strings).
- Returns structured **JSON output**.
- Auto-generated API docs via **Swagger UI**.

---

## 📦 Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- google-generativeai (Gemini SDK)

Install dependencies:

```bash
pip install fastapi uvicorn pydantic google-generativeai
```
▶️ Run the API
Start FastAPI with:

```bash
uvicorn app:app --reload
```

### Swagger Docs available at:

- 👉 http://127.0.0.1:8000/docs
###  ReDoc available at:
- 👉 http://127.0.0.1:8000/redoc


🛠 API Endpoints
POST /extract-fields/
Extracts structured fields from a PDF.

### Request Body

```json
{
  "pdf_path": "path/to/file.pdf",
  "schema": {
    "Policy Number": "string",
    "insured_name": "string",
    "policy_term": "date",
    "total_annual_premium": "number",
    "premium_paid": "number",
    "commission_rate": "string",
    "commission_paid": "number"
  }
}
```
### Response

```json
{
  "Policy Number": "BO211251",
  "insured_name": "ELI & ELIAS SETS LLC",
  "policy_term": "05/25/2025",
  "total_annual_premium": 6532.0,
  "premium_paid": 6532.0,
  "commission_rate": "12.36%",
  "commission_paid": 807.33
}
```
