
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
from enum import Enum
import re
import xmlrpc.client
import json
import uvicorn
import requests
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

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
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("Please set the GOOGLE_API_KEY environment variable with your Gemini API key")

# Enum for CRUD operation types
class OperationType(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

# Pydantic model for Odoo authentication request
class OdooAuthRequest(BaseModel):
    odoo_url: str
    odoo_db: str
    odoo_username: str
    odoo_password: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields

    @classmethod
    def validate_fields(cls, v):
        if not v['odoo_url'] or not v['odoo_db'] or not v['odoo_username'] or not v['odoo_password']:
            raise ValueError("All Odoo credentials (odoo_url, odoo_db, odoo_username, odoo_password) are required")
        return v

# Pydantic model for Odoo CRM lead request
class OdooLeadRequest(BaseModel):
    operation: OperationType
    username: str
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields

    @classmethod
    def validate_fields(cls, v):
        if not v['username']:
            raise ValueError("username is required")
        if v['operation'] in [OperationType.CREATE, OperationType.UPDATE] and not v['data']:
            raise ValueError("data is required for create and update operations")
        if v['operation'] == OperationType.CREATE and (not v['data'] or 'name' not in v['data']):
            raise ValueError("name is required for create operation")
        if v['operation'] == OperationType.UPDATE and (not v['data'] or ('id' not in v['data'] and 'name' not in v['data']) or 'fields' not in v['data']):
            raise ValueError("either id or name, and fields are required for update operation")
        if v['operation'] == OperationType.DELETE and (not v['data'] or ('id' not in v['data'] and 'name' not in v['data'])):
            raise ValueError("data with either id or name is required")
        if v['operation'] == OperationType.DELETE and 'id' in v['data']:
            if isinstance(v['data']['id'], list):
                if not all(isinstance(i, int) for i in v['data']['id']):
                    raise ValueError("id must be an integer or a list of integers")
            elif not isinstance(v['data']['id'], int):
                raise ValueError("id must be an integer or a list of integers")
        return v

# Pydantic model for a single sub-query
class Query(BaseModel):
    id: int = Field(..., description="Unique sub-query identifier")
    question: str = Field(..., description="Sub-query text")
    operation: OperationType = Field(..., description="CRUD operation type")
    lead_name: Optional[str] = Field(None, description="Lead name for create/update/delete")
    lead_id: Optional[int] = Field(None, description="Lead ID for update/delete")
    fields: Optional[Dict[str, Any]] = Field(None, description="Fields for create/update")
    search_term: Optional[str] = Field(None, description="Search term for read")
    limit: Optional[int] = Field(10, description="Limit for read operation")

# Pydantic model for the query plan
class QueryPlan(BaseModel):
    query_graph: List[Query] = Field(..., description="List of sub-queries")

# Pydantic model for the chat request
class ChatRequest(BaseModel):
    query: str
    username: str  # Added required username field

# Odoo Authentication
def authenticate_odoo(username: str) -> Dict[str, Any]:
    try:
        safe_username = username.replace('@', '_').replace('.', '_')
        cred_file = f"odoo_credentials_{safe_username}.json"
        if os.path.exists(cred_file):
            try:
                with open(cred_file, 'r') as f:
                    credentials = json.load(f)
                common = xmlrpc.client.ServerProxy(f"{credentials['odoo_url']}/xmlrpc/2/common")
                uid = common.authenticate(credentials['odoo_db'], credentials['odoo_username'], credentials['odoo_password'], {})
                if uid:
                    models = xmlrpc.client.ServerProxy(f"{credentials['odoo_url']}/xmlrpc/2/object")
                    logger.info(f"Loaded Odoo credentials from {cred_file} for {username}")
                    return {
                        "uid": uid,
                        "models": models,
                        "db": credentials['odoo_db'],
                        "password": credentials['odoo_password'],
                        "status": "success",
                        "error": None
                    }
                else:
                    logger.warning(f"Invalid credentials in {cred_file}, authentication required")
            except Exception as e:
                logger.error(f"Failed to load Odoo credentials from {cred_file}: {str(e)}")

        raise ValueError(f"No valid Odoo credentials found for user {username}. Please authenticate via /auth/odoo.")
    except Exception as e:
        logger.error(f"Odoo authentication failed for {username}: {str(e)}")
        return {
            "uid": None,
            "models": None,
            "db": None,
            "password": None,
            "status": "error",
            "error": f"Odoo authentication failed: {str(e)}"
        }

# Endpoint for Odoo authentication
@app.post("/auth/odoo")
async def auth_odoo(request: OdooAuthRequest):
    try:
        common = xmlrpc.client.ServerProxy(f"{request.odoo_url}/xmlrpc/2/common")
        uid = common.authenticate(request.odoo_db, request.odoo_username, request.odoo_password, {})
        if not uid:
            logger.error(f"Odoo authentication failed for {request.odoo_username}")
            raise HTTPException(status_code=401, detail="Odoo authentication failed: Invalid credentials")

        credentials = {
            "odoo_url": request.odoo_url,
            "odoo_db": request.odoo_db,
            "odoo_username": request.odoo_username,
            "odoo_password": request.odoo_password,
            "uid": uid
        }
        safe_username = request.odoo_username.replace('@', '_').replace('.', '_')
        cred_file = f"odoo_credentials_{safe_username}.json"
        try:
            with open(cred_file, 'w') as f:
                json.dump(credentials, f)
            logger.info(f"Saved Odoo credentials to {cred_file} for {request.odoo_username}")
        except Exception as e:
            logger.error(f"Failed to save Odoo credentials to {cred_file}: {str(e)}")

        logger.info(f"Successfully authenticated Odoo for {request.odoo_username}")
        return {"message": f"Odoo authentication successful for {request.odoo_username}. Use /api/v1/crm/odoo/leads with username='{request.odoo_username}'."}
    except Exception as e:
        logger.error(f"Odoo authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Odoo authentication failed: {str(e)}")

# Endpoint for Odoo CRUD operations
@app.post("/api/v1/crm/odoo/leads")
async def manage_odoo_leads(request: OdooLeadRequest):
    try:
        # Authenticate Odoo
        auth_result = authenticate_odoo(request.username)
        if auth_result['status'] == "error":
            raise HTTPException(status_code=401, detail=auth_result['error'])

        models = auth_result['models']
        db = auth_result['db']
        uid = auth_result['uid']
        password = auth_result['password']

        # Handle CRUD operations
        if request.operation == OperationType.CREATE:
            if not request.data or 'name' not in request.data:
                raise HTTPException(status_code=400, detail="Lead name is required for create operation")
            lead_id = models.execute_kw(db, uid, password, 'crm.lead', 'create', [request.data])
            lead = models.execute_kw(db, uid, password, 'crm.lead', 'read', [[lead_id]])
            logger.info(f"Created lead with ID {lead_id} for {request.username}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": lead[0] if lead else {},
                "message": f"Lead created with ID {lead_id}"
            }

        elif request.operation == OperationType.READ:
            domain = []
            if request.data and 'ids' in request.data:
                domain = [['id', 'in', request.data['ids']]]
            elif request.data and 'search_term' in request.data:
                domain = [['name', 'ilike', request.data['search_term']]]
            leads = models.execute_kw(
                db, uid, password, 'crm.lead', 'search_read',
                [domain], {'limit': request.data.get('limit', 10) if request.data else 10}
            )
            logger.info(f"Retrieved {len(leads)} leads for {request.username}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": leads,
                "message": f"Retrieved {len(leads)} leads"
            }

        elif request.operation == OperationType.UPDATE:
            if not request.data or ('id' not in request.data and 'name' not in request.data) or 'fields' not in request.data:
                raise HTTPException(status_code=400, detail="Either id or name, and fields are required for update operation")
            
            lead_id = None
            if 'id' in request.data:
                lead_id = int(request.data['id'])  # Ensure id is an integer
                # Verify lead exists
                existing_leads = models.execute_kw(
                    db, uid, password, 'crm.lead', 'search',
                    [[['id', '=', lead_id]]]
                )
                if not existing_leads:
                    raise HTTPException(status_code=404, detail=f"No lead found with ID {lead_id}")
            elif 'name' in request.data:
                lead_ids = models.execute_kw(
                    db, uid, password, 'crm.lead', 'search',
                    [[['name', 'ilike', request.data['name']]]], {'limit': 1}
                )
                if not lead_ids:
                    raise HTTPException(status_code=404, detail=f"No lead found with name {request.data['name']}")
                lead_id = lead_ids[0]
            
            fields = request.data['fields']
            models.execute_kw(db, uid, password, 'crm.lead', 'write', [[lead_id], fields])
            lead = models.execute_kw(db, uid, password, 'crm.lead', 'read', [[lead_id]])
            logger.info(f"Updated lead with ID {lead_id} for {request.username}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": lead[0] if lead else {},
                "message": f"Lead with ID {lead_id} updated"
            }

        elif request.operation == OperationType.DELETE:
            if not request.data or ('id' not in request.data and 'name' not in request.data):
                raise HTTPException(status_code=400, detail="Either id or name is required for delete operation")
            
            lead_ids = []
            if 'id' in request.data:
                lead_ids = [int(request.data['id'])] if isinstance(request.data['id'], (int, str)) else [int(i) for i in request.data['id']]
                # Verify leads exist
                existing_leads = models.execute_kw(
                    db, uid, password, 'crm.lead', 'search',
                    [[['id', 'in', lead_ids]]]
                )
                if not existing_leads:
                    raise HTTPException(status_code=404, detail=f"No leads found with IDs {lead_ids}")
                lead_ids = existing_leads  # Only delete existing leads
            elif 'name' in request.data:
                lead_ids = models.execute_kw(
                    db, uid, password, 'crm.lead', 'search',
                    [[['name', 'ilike', request.data['name']]]]
                )
                if not lead_ids:
                    raise HTTPException(status_code=404, detail=f"No leads found with name {request.data['name']}")
            
            models.execute_kw(db, uid, password, 'crm.lead', 'unlink', [lead_ids])
            logger.info(f"Deleted leads with IDs {lead_ids} for {request.username}")
            return {
                "status": "success",
                "operation": "delete",
                "result": [],
                "message": f"Leads with IDs {lead_ids} deleted"
            }

    except Exception as e:
        logger.error(f"Unexpected error for {request.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Load mapping.json
def load_mapping(file_path: str = "mapping.json") -> Dict[str, Any]:
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
    # Prepare operations info
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
    You are an expert at parsing natural language queries for a CRM system. Given the user's query and API mapping, generate a structured JSON output for a single sub-query.

    *User Query*: "{complex_query}"

    *API Mapping*:
    {json.dumps(operations_info, indent=2)}

    *Instructions*:
    - Identify the CRUD operation (create, read, update, delete) by matching the query against the patterns.
    - Extract parameters based on required_fields, optional_fields, identifier, and the example:
      - For 'create': Extract 'lead_name' (required) and optional fields (e.g., phone, email_from).
      - For 'read': Extract 'search_term' and 'limit' (default 10), both optional.
      - For 'update': Extract 'lead_id' (integer) or 'lead_name' (from identifier), and 'fields' (required).
      - For 'delete': Extract 'lead_id' (integer) or 'lead_name' (from identifier).
    - Return a JSON object with:
      - id: 1 (integer)
      - question: The original query (string)
      - operation: The CRUD operation (string: create, read, update, delete)
      - lead_name: Lead name (string, optional)
      - lead_id: Lead ID (integer, optional)
      - fields: Fields dictionary (object, optional)
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
          "lead_name": "<name or null>",
          "lead_id": <id or null>,
          "fields": <fields or null>,
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
    You are a routing assistant for a CRM API. Given a sub-query and API mapping, confirm the CRUD operation.

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
async def handle_crud_operation(sub_query: Query, mapping: Dict[str, Any], username: str) -> Dict[str, Any]:
    """
    Construct the JSON payload and call the internal Odoo CRUD endpoint.

    Args:
        sub_query (Query): The sub-query data with operation and parameters.
        mapping (Dict[str, Any]): The mapping.json content.
        username (str): Odoo username for the API call.

    Returns:
        Dict[str, Any]: The API response or authentication prompt.
    """
    payload = {
        "operation": sub_query.operation.value,
        "username": username,
        "data": {}
    }

    try:
        if sub_query.operation == OperationType.CREATE:
            if not sub_query.lead_name:
                raise ValueError("Lead name is required for create operation")
            payload["data"] = {
                "name": sub_query.lead_name,
                **(sub_query.fields or {})
            }
        elif sub_query.operation == OperationType.READ:
            payload["data"] = {"limit": sub_query.limit}
            if sub_query.search_term:
                payload["data"]["search_term"] = sub_query.search_term
        elif sub_query.operation == OperationType.UPDATE:
            if not sub_query.lead_id and not sub_query.lead_name:
                raise ValueError("Lead ID or name is required for update operation")
            if not sub_query.fields:
                raise ValueError("Fields are required for update operation")
            payload["data"] = {"fields": sub_query.fields}
            if sub_query.lead_id:
                payload["data"]["id"] = sub_query.lead_id
            else:
                payload["data"]["name"] = sub_query.lead_name
        elif sub_query.operation == OperationType.DELETE:
            if not sub_query.lead_id and not sub_query.lead_name:
                raise ValueError("Lead ID or name is required for delete operation")
            if sub_query.lead_id:
                payload["data"]["id"] = sub_query.lead_id
            else:
                payload["data"]["name"] = sub_query.lead_name

        # Call the internal endpoint directly
        request = OdooLeadRequest(**payload)
        response = await manage_odoo_leads(request)
        logger.info(f"Executed {sub_query.operation.value} operation for query: {sub_query.question}")
        return response

    except ValueError as e:
        logger.error(f"Invalid query data: {str(e)}")
        return {"status": "error", "message": str(e)}
    except HTTPException as e:
        logger.error(f"API call failed: {str(e)}")
        return {"status": "error", "message": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# Main agent function
async def process_query(query: str, username: str) -> Dict[str, Any]:
    """
    Process a user query and execute the corresponding CRUD operation.

    Args:
        query (str): The user's natural language query.
        username (str): Odoo username for the API call.

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
            result = await handle_crud_operation(sub_query, mapping, username)
            return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"status": "error", "message": str(e)}

# Chat endpoint
@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """
    Process a natural language query and return the CRUD operation result.

    Args:
        request (ChatRequest): JSON body with 'query' and 'username' fields.

    Returns:
        Dict[str, Any]: The API response or error message.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    
    logger.info(f"Received query: {request.query} for user: {request.username}")
    response = await process_query(request.query, request.username)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)