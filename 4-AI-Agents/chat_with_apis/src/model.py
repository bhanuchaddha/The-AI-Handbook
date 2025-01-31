from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class APIEndpoint(BaseModel):
    path: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method")
    description: str = Field(..., description="Description of the endpoint")
    parameters: Dict = Field(default={}, description="API parameters")
    responses: Dict = Field(default={}, description="Expected responses")

class APIDocument(BaseModel):
    title: str = Field(..., description="Title of the API documentation")
    base_url: str = Field(..., description="Base URL for the API")
    endpoints: List[APIEndpoint] = Field(default=[], description="List of API endpoints")
    authentication: Dict = Field(default={}, description="Authentication details")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Message content")
    api_call: Optional[Dict] = Field(default=None, description="API call details if any")
