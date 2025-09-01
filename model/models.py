from pydantic import BaseModel
from typing import Dict, Any

# Request model
class ExtractRequest(BaseModel):
    pdf_path: str
    schema: Dict[str, Any]
    total_schema:Dict[str, Any]

class ExtractHeader(BaseModel):
    pdf_path: str
    schema: Dict[str, Any]
