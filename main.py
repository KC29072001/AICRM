# saves odoo credentials   
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from typing import Optional, List, TypedDict, Literal, Dict, Any
import json
import google.generativeai as genai
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import re
import logging
import xmlrpc.client  # For Odoo API integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# In-memory storage for credentials (use a database in production)
CREDENTIALS_STORAGE = {}

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

# Pydantic model for retrieve request
class LocateRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    gmail_address: Optional[str] = None
    source: Literal["gmail", "odoo"]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields

    @classmethod
    def validate_fields(cls, v):
        if v['source'] == "gmail" and not v['gmail_address']:
            raise ValueError("gmail_address is required for source 'gmail'")
        if v['gmail_address'] and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v['gmail_address']):
            raise ValueError("gmail_address must be a valid email address")
        return v

# Gmail API credential setup
def get_gmail_credentials(gmail_address: str):
    creds = CREDENTIALS_STORAGE.get(gmail_address)
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']

    # Try loading from file if not in memory
    if not creds:
        safe_email = gmail_address.replace('@', '_').replace('.', '_')
        token_file = f"token_{safe_email}.json"
        if os.path.exists(token_file):
            try:
                creds = Credentials.from_authorized_user_file(token_file, scopes)
                CREDENTIALS_STORAGE[gmail_address] = creds
                logger.info(f"Loaded credentials from {token_file} for {gmail_address}")
            except Exception as e:
                logger.error(f"Failed to load credentials from {token_file}: {str(e)}")
                creds = None

    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                logger.info(f"Refreshing credentials for {gmail_address}")
                creds.refresh(GoogleRequest())
                CREDENTIALS_STORAGE[gmail_address] = creds
                safe_email = gmail_address.replace('@', '_').replace('.', '_')
                token_file = f"token_{safe_email}.json"
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved refreshed credentials to {token_file}")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {str(e)}")
                raise RuntimeError(f"Failed to refresh credentials: {str(e)}")
        else:
            creds = None

    if not creds:
        raise HTTPException(
            status_code=401,
            detail=f"No valid credentials for {gmail_address}. Please authenticate via /auth/start."
        )

    return creds

# OAuth flow configuration
def get_oauth_flow(redirect_uri: str):
    credentials_path = "credentials.json"
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Google API credentials file not found at {credentials_path}")

    try:
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/gmail.readonly'],
            redirect_uri=redirect_uri
        )
        return flow
    except Exception as e:
        logger.error(f"Failed to initialize OAuth flow: {str(e)}")
        raise RuntimeError(f"Failed to initialize OAuth flow: {str(e)}")

# Endpoint to start OAuth flow
@app.get("/auth/start")
async def auth_start(gmail_address: str, request: FastAPIRequest):
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", gmail_address):
        raise HTTPException(status_code=400, detail="Invalid gmail_address")

    safe_email = gmail_address.replace('@', '_').replace('.', '_')
    token_file = f"token_{safe_email}.json"
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, scopes)
            if creds.valid:
                logger.info(f"Valid credentials found for {gmail_address} in {token_file}")
                return {"message": f"Already authenticated for {gmail_address}. Use /api/v1/retrieve directly."}
            elif creds.expired and creds.refresh_token:
                logger.info(f"Refreshing expired credentials for {gmail_address}")
                creds.refresh(GoogleRequest())
                CREDENTIALS_STORAGE[gmail_address] = creds
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved refreshed credentials to {token_file}")
                return {"message": f"Already authenticated for {gmail_address}. Use /api/v1/retrieve directly."}
        except Exception as e:
            logger.error(f"Failed to load or refresh credentials from {token_file}: {str(e)}")
            creds = None

    redirect_uri = f"{request.base_url}auth/callback"
    flow = get_oauth_flow(redirect_uri)
    auth_url, state = flow.authorization_url(prompt='consent', access_type='offline')
    CREDENTIALS_STORAGE[f"{gmail_address}_state"] = state
    logger.info(f"Generated auth URL for {gmail_address} with redirect_uri: {redirect_uri}")
    return {"auth_url": auth_url}

# Endpoint to handle OAuth callback
@app.get("/auth/callback")
async def auth_callback(request: FastAPIRequest):
    state = request.query_params.get('state')
    code = request.query_params.get('code')
    if not state or not code:
        logger.error("Invalid callback parameters")
        raise HTTPException(status_code=400, detail="Invalid callback parameters")

    gmail_address = None
    for key, stored_state in CREDENTIALS_STORAGE.items():
        if stored_state == state and key.endswith("_state"):
            gmail_address = key.replace("_state", "")
            break

    if not gmail_address:
        logger.error("Invalid state in callback")
        raise HTTPException(status_code=400, detail="Invalid state")

    redirect_uri = f"{request.base_url}auth/callback"
    flow = get_oauth_flow(redirect_uri)
    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        CREDENTIALS_STORAGE[gmail_address] = creds
        safe_email = gmail_address.replace('@', '_').replace('.', '_')
        token_file = f"token_{safe_email}.json"
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        logger.info(f"Saved credentials to {token_file} for {gmail_address}")
        del CREDENTIALS_STORAGE[f"{gmail_address}_state"]
        logger.info(f"Successfully stored credentials for {gmail_address}")
        return RedirectResponse(url="/auth/success")
    except Exception as e:
        logger.error(f"Failed to fetch token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch token: {str(e)}")

# Endpoint for successful Gmail authentication
@app.get("/auth/success")
async def auth_success():
    return {"message": "Gmail authentication successful. You can now use the /api/v1/retrieve endpoint with source='gmail'."}

# Endpoint for Odoo authentication
@app.post("/auth/odoo")
async def auth_odoo(request: OdooAuthRequest):
    try:
        # Attempt to authenticate with provided credentials
        common = xmlrpc.client.ServerProxy(f"{request.odoo_url}/xmlrpc/2/common")
        uid = common.authenticate(request.odoo_db, request.odoo_username, request.odoo_password, {})
        if not uid:
            logger.error(f"Odoo authentication failed for {request.odoo_username}")
            raise HTTPException(status_code=401, detail="Odoo authentication failed: Invalid credentials")

        # Store credentials in memory
        CREDENTIALS_STORAGE[f"odoo_{request.odoo_username}"] = {
            "odoo_url": request.odoo_url,
            "odoo_db": request.odoo_db,
            "odoo_username": request.odoo_username,
            "odoo_password": request.odoo_password,
            "uid": uid
        }

        # Save credentials to file
        safe_username = request.odoo_username.replace('@', '_').replace('.', '_')
        cred_file = f"odoo_credentials_{safe_username}.json"
        try:
            with open(cred_file, 'w') as f:
                json.dump(CREDENTIALS_STORAGE[f"odoo_{request.odoo_username}"], f)
            logger.info(f"Saved Odoo credentials to {cred_file} for {request.odoo_username}")
        except Exception as e:
            logger.error(f"Failed to save Odoo credentials to {cred_file}: {str(e)}")

        logger.info(f"Successfully authenticated Odoo for {request.odoo_username}")
        return {"message": f"Odoo authentication successful for {request.odoo_username}. Use /api/v1/retrieve with source='odoo' and gmail_address='{request.odoo_username}'."}
    except Exception as e:
        logger.error(f"Odoo authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Odoo authentication failed: {str(e)}")

# Odoo Authentication
def authenticate_odoo(user_id: str) -> Dict[str, Any]:
    try:
        # Check for existing credentials file
        safe_username = user_id.replace('@', '_').replace('.', '_')
        cred_file = f"odoo_credentials_{safe_username}.json"
        if os.path.exists(cred_file):
            try:
                with open(cred_file, 'r') as f:
                    credentials = json.load(f)
                # Verify credentials by attempting authentication
                common = xmlrpc.client.ServerProxy(f"{credentials['odoo_url']}/xmlrpc/2/common")
                uid = common.authenticate(credentials['odoo_db'], credentials['odoo_username'], credentials['odoo_password'], {})
                if uid:
                    models = xmlrpc.client.ServerProxy(f"{credentials['odoo_url']}/xmlrpc/2/object")
                    CREDENTIALS_STORAGE[f"odoo_{user_id}"] = credentials
                    logger.info(f"Loaded Odoo credentials from {cred_file} for {user_id}")
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

        # Fall back to in-memory credentials
        credentials = CREDENTIALS_STORAGE.get(f"odoo_{user_id}")
        if not credentials:
            raise ValueError(f"No Odoo credentials found for user {user_id}. Please authenticate via /auth/odoo.")

        odoo_url = credentials["odoo_url"]
        db = credentials["odoo_db"]
        username = credentials["odoo_username"]
        password = credentials["odoo_password"]

        common = xmlrpc.client.ServerProxy(f'{odoo_url}/xmlrpc/2/common')
        uid = common.authenticate(db, username, password, {})
        if not uid:
            raise ValueError("Odoo authentication failed")

        models = xmlrpc.client.ServerProxy(f'{odoo_url}/xmlrpc/2/object')
        return {
            "uid": uid,
            "models": models,
            "db": db,
            "password": password,
            "status": "success",
            "error": None
        }
    except Exception as e:
        logger.error(f"Odoo authentication failed for {user_id}: {str(e)}")
        return {
            "uid": None,
            "models": None,
            "db": None,
            "password": None,
            "status": "error",
            "error": f"Odoo authentication failed: {str(e)}"
        }

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise RuntimeError(f"Failed to initialize Gemini API: {str(e)}")

# Define the state for LangGraph
class AgentState(TypedDict):
    query: str
    max_results: int
    gmail_address: Optional[str]
    messages: List[dict]
    leads: List[dict]
    odoo_auth: Dict[str, Any]
    source: Literal["gmail", "odoo"]
    summary: str
    status: str
    error: Optional[str]

# Query Router Node
def query_router(state: AgentState) -> Command[Literal["search_emails", "odoo_authenticate", "__end__"]]:
    if state['source'] == "odoo":
        return Command(
            update={
                "status": "pending",
                "error": None
            },
            goto="odoo_authenticate"
        )
    return Command(
        update={
            "status": "pending",
            "error": None
        },
        goto="search_emails"
    )

# Odoo Authenticate Node
def odoo_authenticate(state: AgentState) -> Command[Literal["search_leads", "__end__"]]:
    auth_result = authenticate_odoo(state['gmail_address'])
    if auth_result['status'] == "error":
        return Command(
            update={
                "status": "error",
                "error": auth_result['error']
            },
            goto="__end__"
        )
    return Command(
        update={
            "odoo_auth": auth_result,
            "status": "success",
            "error": None
        },
        goto="search_leads"
    )

# Search Leads Node (Odoo)
def search_leads(state: AgentState) -> Command[Literal["summarizer", "__end__"]]:
    try:
        odoo_auth = state['odoo_auth']
        if odoo_auth['status'] != "success":
            return Command(
                update={
                    "status": "error",
                    "error": "Odoo authentication not successful"
                },
                goto="__end__"
            )

        search_term = state['query'].strip()
        if not search_term:
            return Command(
                update={
                    "leads": [],
                    "status": "success",
                    "error": "No search term provided for leads"
                },
                goto="summarizer"
            )

        # Search leads using Odoo API (search_read)
        models = odoo_auth['models']
        leads = models.execute_kw(
            odoo_auth['db'], odoo_auth['uid'], odoo_auth['password'],
            'crm.lead', 'search_read',
            [[['name', 'ilike', search_term]]],
            {'limit': state['max_results']}
        )

        if not leads:
            return Command(
                update={
                    "leads": [],
                    "status": "success",
                    "error": "No leads found"
                },
                goto="summarizer"
            )

        return Command(
            update={
                "leads": leads,
                "status": "success",
                "error": None
            },
            goto="summarizer"
        )
    except Exception as e:
        logger.error(f"Error searching leads: {str(e)}")
        return Command(
            update={
                "status": "error",
                "error": f"Error searching leads: {str(e)}"
            },
            goto="__end__"
        )

# Email Search Node
def search_emails(state: AgentState) -> Command[Literal["summarizer", "__end__"]]:
    try:
        creds = get_gmail_credentials(state['gmail_address'])
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(
            userId=state['gmail_address'],
            q=state['query'],
            maxResults=state['max_results']
        ).execute()
        messages = results.get('messages', [])
        message_details = []
        for msg in messages:
            msg_data = service.users().messages().get(userId=state['gmail_address'], id=msg['id']).execute()
            headers = msg_data.get('payload', {}).get('headers', [])
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
            message_details.append({
                'id': msg['id'],
                'snippet': msg_data.get('snippet', ''),
                'from': sender,
                'subject': subject,
                'date': msg_data.get('internalDate', ''),
                'link': f"https://mail.google.com/mail/u/0/#inbox/{msg['id']}"
            })
        if not message_details:
            return Command(
                update={
                    "messages": [],
                    "status": "success",
                    "error": "No results found"
                },
                goto="__end__"
            )
        return Command(
            update={
                "messages": message_details,
                "status": "success",
                "error": None
            },
            goto="summarizer"
        )
    except Exception as e:
        logger.error(f"Error searching emails for {state['gmail_address']}: {str(e)}")
        return Command(
            update={
                "status": "error",
                "error": f"Error searching emails: {str(e)}"
            },
            goto="__end__"
        )

# Summarizer Node
def summarize_results(state: AgentState) -> Command[Literal["__end__"]]:
    try:
        if state['source'] == "odoo":
            prompt = f"""
            Summarize the following Odoo lead search results for the query '{state['query']}' in a concise manner (max 200 words).
            Focus on key information relevant to the query, such as lead names, partners, probabilities, email, phone, salesperson, expected revenue and stages if available.

            Search Results:
            {json.dumps(state['leads'], indent=2)}

            Provide a clear, professional summary:
            """
        else:
            prompt = f"""
            Summarize the following email search results for the query '{state['query']}' in a concise manner (max 200 words).
            Focus on key information relevant to the query. Do not include sensitive personal information.

            Search Results:
            {json.dumps(state['messages'], indent=2)}

            Provide a clear, professional summary:
            """

        response = gemini_model.generate_content(prompt)
        summary = response.text.strip()

        return Command(
            update={
                "summary": summary,
                "status": "success",
                "error": None
            },
            goto="__end__"
        )
    except Exception as e:
        logger.error(f"Error summarizing results: {str(e)}")
        return Command(
            update={
                "status": "error",
                "error": f"Error summarizing results: {str(e)}"
            },
            goto="__end__"
        )

# Define the Updated LangGraph Workflow
def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("query_router", query_router)
    workflow.add_node("odoo_authenticate", odoo_authenticate)
    workflow.add_node("search_leads", search_leads)
    workflow.add_node("search_emails", search_emails)
    workflow.add_node("summarizer", summarize_results)

    # Define edges
    workflow.add_edge(START, "query_router")
    workflow.add_conditional_edges(
        "query_router",
        lambda state: state.get("goto", "__end__"),
        {
            "search_emails": "search_emails",
            "odoo_authenticate": "odoo_authenticate",
            "__end__": END
        }
    )
    workflow.add_conditional_edges(
        "odoo_authenticate",
        lambda state: state.get("goto", "__end__"),
        {
            "search_leads": "search_leads",
            "__end__": END
        }
    )
    workflow.add_conditional_edges(
        "search_leads",
        lambda state: state.get("goto", "__end__"),
        {
            "summarizer": "summarizer",
            "__end__": END
        }
    )
    workflow.add_conditional_edges(
        "search_emails",
        lambda state: state.get("goto", "__end__"),
        {
            "summarizer": "summarizer",
            "__end__": END
        }
    )
    workflow.add_edge("summarizer", END)

    return workflow.compile()

# Initialize the graph
graph = create_workflow()

@app.post("/api/v1/retrieve")
async def retrieve_info(request: LocateRequest):
    """Endpoint to retrieve and summarize information from Gmail or Odoo"""
    try:
        initial_state = AgentState(
            query=request.query,
            max_results=request.max_results,
            gmail_address=request.gmail_address,
            messages=[],
            leads=[],
            odoo_auth={},
            source=request.source,
            summary="",
            status="pending",
            error=None
        )

        final_state = graph.invoke(initial_state)

        if final_state['status'] == "error":
            raise HTTPException(status_code=500, detail=final_state['error'])

        logger.info(f"Successfully processed request for source {request.source}")
        return {
            "status": final_state['status'],
            "results": final_state['messages'] if final_state['source'] == "gmail" else final_state['leads'],
            "source": final_state['source'],
            "summary": final_state['summary']
        }
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)