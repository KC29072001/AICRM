from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import uvicorn
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from gmail_crud import manage_leads, LeadRequest, OperationType, get_gmail_credentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini LLM
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
except KeyError:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("Please set the GEMINI_API_KEY environment variable with your Gemini API key")

# Pydantic model for a single sub-query
class Query(BaseModel):
    id: int = Field(..., description="Unique sub-query identifier")
    question: str = Field(..., description="Sub-query text")
    operation: OperationType = Field(..., description="CRUD operation type")
    message_id: Optional[str] = Field(None, description="Message ID for update/delete")
    to: Optional[str] = Field(None, description="Recipient email for create/update")
    subject: Optional[str] = Field(None, description="Subject for create/update")
    body: Optional[str] = Field(None, description="Body for create/update")
    search_term: Optional[str] = Field(None, description="Search term for read")
    limit: Optional[int] = Field(10, description="Limit for read operation")

# Pydantic model for the query plan
class QueryPlan(BaseModel):
    query_graph: List[Query] = Field(..., description="List of sub-queries")

# Pydantic model for the chat request
class ChatRequest(BaseModel):
    query: str
    gmail_address: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields

    @classmethod
    def validate_fields(cls, v):
        if not v['query'] or not v['gmail_address']:
            raise ValueError("query and gmail_address are required")
        if not isinstance(v['gmail_address'], str) or '@' not in v['gmail_address']:
            raise ValueError("gmail_address must be a valid email address")
        return v

# Load mapping.json
def load_mapping(file_path: str = "gmail_mapping.json") -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mapping.json: {str(e)}")
        raise ValueError(f"Failed to load mapping.json: {str(e)}")

# Query Planner using Gemini LLM
def plan_query(complex_query: str, mapping: Dict[str, Any]) -> QueryPlan:
    """
    Parse a user query into a structured QueryPlan using Gemini LLM and mapping.json.

    Args:
        complex_query (str): The user's natural language query.
        mapping (Dict[str, Any]): The mapping.json content.

    Returns:
        QueryPlan: Structured plan with operation and parameters.
    """
    operations_info = []
    for op, details in mapping["operations"].items():
        operations_info.append({
            "operation": op,
            "patterns": details["patterns"],
            "required_fields": details.get("required_fields", []),
            "optional_fields": details.get("optional_fields", []),
            "identifier": details.get("identifier", []),
            "example": details["example"]
        })

    prompt = f"""
    You are an expert at parsing natural language queries for a Gmail API system. Given the user's query and API mapping, generate a structured JSON output for a single sub-query.

    *User Query*: "{complex_query}"

    *API Mapping*:
    {json.dumps(operations_info, indent=2)}

    *Instructions*:
    - Identify the CRUD operation (create, read, update, delete) by matching the query against the patterns.
    - Extract parameters based on required_fields, optional_fields, identifier, and the example:
      - For 'create': Extract 'to', 'subject', 'body' (all required).
      - For 'read': Extract 'search_term' and 'limit' (default 10), both optional.
      - For 'update': Extract 'message_id' (required) and 'to', 'subject', 'body' (all required).
      - For 'delete': Extract 'message_id' (required, can be a single ID or list of IDs).
    - Return a JSON object with:
      - id: 1 (integer)
      - question: The original query (string)
      - operation: The CRUD operation (string: create, read, update, delete)
      - message_id: Message ID (string, optional)
      - to: Recipient email (string, optional)
      - subject: Subject (string, optional)
      - body: Body (string, optional)
      - search_term: Search term for read (string, optional)
      - limit: Limit for read (integer, default 10)
    - If the query is invalid or doesn't match any operation, include an error message.

    *Output Format*:
    ```json
    {{
      "query_graph": [
        {{
          "id": 1,
          "question": "<query>",
          "operation": "<operation>",
          "message_id": "<id or null>",
          "to": "<to or null>",
          "subject": "<subject or null>",
          "body": "<body or null>",
          "search_term": "<term or null>",
          "limit": <limit or 10>
        }}
      ]
    }}
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        logger.debug(f"Gemini raw response: {response.text}")
        result = json.loads(response.text)
        return QueryPlan.model_validate(result)
    except Exception as e:
        logger.error(f"Gemini LLM query planning failed: {str(e)}")
        raise ValueError(f"Failed to parse query: {str(e)}")

# Router using Gemini LLM
def llm_router(sub_query: Query, mapping: Dict[str, Any]) -> str:
    """
    Route the sub-query to the appropriate CRUD operation using Gemini LLM.

    Args:
        sub_query (Query): The sub-query data.
        mapping (Dict[str, Any]): The mapping.json content.

    Returns:
        str: The operation type.
    """
    operations = list(mapping["operations"].keys())
    prompt = f"""
    You are a routing assistant for a Gmail API. Given a sub-query and API mapping, confirm the CRUD operation.

    *Sub-Query*:
    ```json
    {sub_query.model_dump_json()}
    ```

    *API Mapping Operations*:
    {json.dumps(operations, indent=2)}

    *Instructions*:
    - Validate the sub-query's operation field against the available operations.
    - If the operation is valid, return it as a string (create, read, update, or delete).
    - If invalid, throw an error with a descriptive message.

    *Output Format*:
    ```json
    "<operation>"
    ```
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        operation = json.loads(response.text)
        if operation not in operations:
            raise ValueError(f"Invalid operation: {operation}")
        return operation
    except Exception as e:
        logger.error(f"Gemini LLM routing failed: {str(e)}")
        raise ValueError(f"Routing failed: {str(e)}")

# Tool to execute the API call
async def handle_crud_operation(sub_query: Query, mapping: Dict[str, Any], gmail_address: str) -> Dict[str, Any]:
    """
    Construct the JSON payload and call the internal Gmail CRUD endpoint.

    Args:
        sub_query (Query): The sub-query data with operation and parameters.
        mapping (Dict[str, Any]): The mapping.json content.
        gmail_address (str): Gmail address for the API call.

    Returns:
        Dict[str, Any]: The API response or authentication prompt.
    """
    payload = {
        "operation": sub_query.operation.value,
        "gmail_address": gmail_address,
        "data": {}
    }

    try:
        # Verify credentials exist
        get_gmail_credentials(gmail_address)

        if sub_query.operation == OperationType.CREATE:
            if not sub_query.to or not sub_query.subject or not sub_query.body:
                raise ValueError("to, subject, and body are required for create operation")
            payload["data"] = {
                "to": sub_query.to,
                "subject": sub_query.subject,
                "body": sub_query.body
            }
        elif sub_query.operation == OperationType.READ:
            payload["data"] = {"limit": sub_query.limit}
            if sub_query.search_term:
                payload["data"]["search_term"] = sub_query.search_term
        elif sub_query.operation == OperationType.UPDATE:
            if not sub_query.message_id or not sub_query.to or not sub_query.subject or not sub_query.body:
                raise ValueError("message_id, to, subject, and body are required for update operation")
            payload["data"] = {
                "id": sub_query.message_id,
                "to": sub_query.to,
                "subject": sub_query.subject,
                "body": sub_query.body,
                "delete_original": True
            }
        elif sub_query.operation == OperationType.DELETE:
            if not sub_query.message_id:
                raise ValueError("message_id is required for delete operation")
            payload["data"] = {"ids": [sub_query.message_id] if isinstance(sub_query.message_id, str) else sub_query.message_id}

        # Call the internal endpoint from gmail-crud.py
        request = LeadRequest(**payload)
        response = await manage_leads(request)
        logger.info(f"Executed {sub_query.operation.value} operation for query: {sub_query.question}")
        return response

    except ValueError as e:
        logger.error(f"Invalid query data: {str(e)}")
        return {"status": "error", "message": str(e)}
    except HTTPException as e:
        logger.error(f"API call failed: {str(e)}")
        if e.status_code == 401:
            return {
                "status": "error",
                "message": f"Please authenticate via /auth/start for {gmail_address}"
            }
        return {"status": "error", "message": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# Main agent function
async def process_query(query: str, gmail_address: str) -> Dict[str, Any]:
    """
    Process a user query and execute the corresponding CRUD operation.

    Args:
        query (str): The user's natural language query.
        gmail_address (str): Gmail address for the API call.

    Returns:
        Dict[str, Any]: The API response or error message.
    """
    try:
        # Load mapping.json
        mapping = load_mapping()

        # Step 1: Plan the query
        plan = plan_query(query, mapping)
        if not plan.query_graph:
            return {"status": "error", "message": "No valid sub-queries found"}

        # Step 2: Process each sub-query (single query for now)
        for sub_query in plan.query_graph:
            # Step 3: Route to the appropriate operation
            operation = llm_router(sub_query, mapping)
            if operation not in mapping["operations"]:
                return {"status": "error", "message": f"Invalid operation: {operation}"}

            # Step 4: Execute the API call
            result = await handle_crud_operation(sub_query, mapping, gmail_address)
            return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"status": "error", "message": str(e)}

# Chat endpoint
@app.post("/gmail/api/v1/chat")
async def chat(request: ChatRequest):
    """
    Process a natural language query and return the CRUD operation result.

    Args:
        request (ChatRequest): JSON body with 'query' and 'gmail_address' fields.

    Returns:
        Dict[str, Any]: The API response or error message.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not request.gmail_address.strip():
        raise HTTPException(status_code=400, detail="gmail_address cannot be empty")
    
    logger.info(f"Received query: {request.query} for Gmail address: {request.gmail_address}")
    response = await process_query(request.query, request.gmail_address)
    return response

# Note: Authentication endpoints are provided by info-ret-agent.py
# Use /auth/start and /auth/callback for Gmail OAuth authentication

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)