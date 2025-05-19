# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# import requests
# from typing import Optional, List, TypedDict, Literal
# import json
# import google.generativeai as genai
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Command

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # Pydantic model for request body
# class LocateRequest(BaseModel):
#     query: str
#     max_results: Optional[int] = 10
#     recipient_phone: str

# # Gmail API credential setup
# def get_gmail_credentials():
#     creds = None
#     token_path = "token.json"
#     credentials_path = "credentials.json"
#     scopes = ['https://www.googleapis.com/auth/gmail.readonly']

#     if os.path.exists(token_path):
#         creds = Credentials.from_authorized_user_file(token_path, scopes)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 credentials_path,
#                 scopes
#             )
#             creds = flow.run_local_server(port=0)
#             with open(token_path, 'w') as token:
#                 token.write(creds.to_json())

#     return creds

# # Initialize Gemini
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# # WhatsApp API configuration
# WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
# WHATSAPP_HEADERS = {
#     "Authorization": f"Bearer {os.getenv('WHATSAPP_ACCESS_TOKEN')}",
#     "Content-Type": "application/json"
# }

# # Define the state for LangGraph
# class AgentState(TypedDict):
#     query: str
#     max_results: int
#     recipient_phone: str
#     messages: List[dict]
#     summary: str
#     whatsapp_status: dict
#     status: str
#     error: Optional[str]


# # Email Search Node
# def search_emails(state: AgentState) -> Command[Literal["summarizer", "__end__"]]:
#     try:
#         creds = get_gmail_credentials()
#         service = build('gmail', 'v1', credentials=creds)
#         results = service.users().messages().list(userId='me', q=state['query'], maxResults=state['max_results']).execute()
#         messages = results.get('messages', [])
#         message_details = []
#         for msg in messages:
#             msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
#             headers = msg_data['payload']['headers']
#             subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
#             sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
#             message_details.append({
#                 'id': msg['id'],
#                 'snippet': msg_data['snippet'],
#                 'from': sender,
#                 'subject': subject,
#                 'date': msg_data.get('internalDate')
#             })
        
#         if not message_details:
#             return Command(
#                 update={
#                     "messages": [],
#                     "status": "success",
#                     "error": "No results found"
#                 },
#                 goto="__end__"
#             )
        
#         return Command(
#             update={
#                 "messages": message_details,
#                 "status": "success"
#             },
#             goto="summarizer"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error searching emails: {str(e)}"
#             },
#             goto="__end__"
#         )

# # Summarizer Node
# def summarize_results(state: AgentState) -> Command[Literal["whatsapp", "__end__"]]:
#     try:
#         prompt = f"""
#         Summarize the following email search results for the query '{state['query']}' in a concise manner (max 200 words).
#         Focus on key information relevant to the query. Do not include sensitive personal information.
        
#         Search Results:
#         {json.dumps(state['messages'], indent=2)}
        
#         Provide a clear, professional summary:
#         """
        
#         response = gemini_model.generate_content(prompt)
#         summary = response.text.strip()
        
#         return Command(
#             update={
#                 "summary": summary,
#                 "status": "success"
#             },
#             goto="whatsapp"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error summarizing results: {str(e)}"
#             },
#             goto="__end__"
#         )

# # WhatsApp Node
# def send_whatsapp_message(state: AgentState) -> Command[Literal["__end__"]]:
#     try:
#         payload = {
#             "messaging_product": "whatsapp",
#             "to": state['recipient_phone'],
#             "type": "text",
#             "text": {"body": state['summary']}
#         }
        
#         response = requests.post(
#             WHATSAPP_API_URL,
#             headers=WHATSAPP_HEADERS,
#             json=payload
#         )
#         response.raise_for_status()
        
#         return Command(
#             update={
#                 "whatsapp_status": {"status": "Message sent successfully"},
#                 "status": "success"
#             },
#             goto="__end__"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error sending WhatsApp message: {str(e)}"
#             },
#             goto="__end__"
#         )

# # Define the LangGraph workflow
# def create_workflow():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("searcher", search_emails)
#     workflow.add_node("summarizer", summarize_results)
#     workflow.add_node("whatsapp", send_whatsapp_message)
    
#     workflow.add_edge(START, "searcher")
#     workflow.add_edge("searcher", "summarizer")
#     workflow.add_edge("summarizer", "whatsapp")
    
#     return workflow.compile()

# # Initialize the graph
# graph = create_workflow()

# @app.post("/api/v1/retrieve")
# async def retrieve_info(request: LocateRequest):
#     """Endpoint to retrieve and summarize information from Gmail and send via WhatsApp"""
#     try:
#         # Initialize state
#         initial_state = AgentState(
#             query=request.query,
#             max_results=request.max_results,
#             recipient_phone=request.recipient_phone,
#             messages=[],
#             summary="",
#             whatsapp_status={},
#             status="pending",
#             error=None
#         )
        
#         # Run the graph
#         final_state = graph.invoke(initial_state)
        
#         if final_state['status'] == "error":
#             raise HTTPException(status_code=500, detail=final_state['error'])
        
#         return {
#             "status": final_state['status'],
#             "results": final_state['messages'],
#             "summary": final_state['summary'],
#             "whatsapp_status": final_state['whatsapp_status']
#         }
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# # Updated code with gmail input in request
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# import requests
# from typing import Optional, List, TypedDict, Literal
# import json
# import google.generativeai as genai
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Command
# import re

# # Load environment variables
# load_dotenv()

# # Validate environment variables
# required_env_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "WHATSAPP_API_KEY", "WHATSAPP_ACCESS_TOKEN", "PHONE_NUMBER_ID"]
# missing_vars = [var for var in required_env_vars if not os.getenv(var)]
# if missing_vars:
#     raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

# app = FastAPI()

# # Pydantic model for request body
# class LocateRequest(BaseModel):
#     query: str
#     max_results: Optional[int] = 10
#     recipient_phone: str
#     gmail_address: str

#     # Validate phone number format (international, digits only, 10-15 characters)
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate_phone

#     @classmethod
#     def validate_phone(cls, v):
#         if not re.match(r"^\d{10,15}$", v['recipient_phone']):
#             raise ValueError("recipient_phone must be 10-15 digits (international format, no '+')")
#         return v

# # Gmail API credential setup
# def get_gmail_credentials(gmail_address: str):
#     creds = None
#     token_path = f"token_{gmail_address.replace('@', '_')}.json"
#     credentials_path = "credentials.json"
#     scopes = ['https://www.googleapis.com/auth/gmail.readonly']

#     if not os.path.exists(credentials_path):
#         raise FileNotFoundError(f"Google API credentials file not found at {credentials_path}")

#     if os.path.exists(token_path):
#         try:
#             creds = Credentials.from_authorized_user_file(token_path, scopes)
#         except Exception as e:
#             raise ValueError(f"Invalid token file {token_path}: {str(e)}")

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             try:
#                 creds.refresh(Request())
#             except Exception as e:
#                 raise RuntimeError(f"Failed to refresh credentials: {str(e)}")
#         else:
#             try:
#                 flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
#                 creds = flow.run_local_server(port=0)
#                 with open(token_path, 'w') as token:
#                     token.write(creds.to_json())
#             except Exception as e:
#                 raise RuntimeError(f"Failed to generate new credentials: {str(e)}")

#     return creds

# # Initialize Gemini
# try:
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     gemini_model = genai.GenerativeModel('gemini-1.5-flash')
# except Exception as e:
#     raise RuntimeError(f"Failed to initialize Gemini API: {str(e)}")

# # WhatsApp API configuration
# WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
# WHATSAPP_HEADERS = {
#     "Authorization": f"Bearer {os.getenv('WHATSAPP_ACCESS_TOKEN')}",
#     "Content-Type": "application/json"
# }

# # Define the state for LangGraph
# class AgentState(TypedDict):
#     query: str
#     max_results: int
#     recipient_phone: str
#     gmail_address: str
#     messages: List[dict]
#     summary: str
#     whatsapp_status: dict
#     status: str
#     error: Optional[str]

# # Email Search Node
# def search_emails(state: AgentState) -> Command[Literal["summarizer", "__end__"]]:
#     try:
#         creds = get_gmail_credentials(state['gmail_address'])
#         service = build('gmail', 'v1', credentials=creds)
#         results = service.users().messages().list(
#             userId=state['gmail_address'],
#             q=state['query'],
#             maxResults=state['max_results']
#         ).execute()
#         messages = results.get('messages', [])
#         message_details = []
#         for msg in messages:
#             msg_data = service.users().messages().get(userId=state['gmail_address'], id=msg['id']).execute()
#             headers = msg_data.get('payload', {}).get('headers', [])
#             subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
#             sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
#             message_details.append({
#                 'id': msg['id'],
#                 'snippet': msg_data.get('snippet', ''),
#                 'from': sender,
#                 'subject': subject,
#                 'date': msg_data.get('internalDate', '')
#             })

#         if not message_details:
#             return Command(
#                 update={
#                     "messages": [],
#                     "status": "success",
#                     "error": "No results found"
#                 },
#                 goto="__end__"
#             )

#         return Command(
#             update={
#                 "messages": message_details,
#                 "status": "success",
#                 "error": None
#             },
#             goto="summarizer"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error searching emails: {str(e)}"
#             },
#             goto="__end__"
#         )

# # Summarizer Node
# def summarize_results(state: AgentState) -> Command[Literal["whatsapp", "__end__"]]:
#     try:
#         prompt = f"""
#         Summarize the following email search results for the query '{state['query']}' in a concise manner (max 200 words).
#         Focus on key information relevant to the query. Do not include sensitive personal information.

#         Search Results:
#         {json.dumps(state['messages'], indent=2)}

#         Provide a clear, professional summary:
#         """

#         response = gemini_model.generate_content(prompt)
#         summary = response.text.strip()

#         return Command(
#             update={
#                 "summary": summary,
#                 "status": "success",
#                 "error": None
#             },
#             goto="whatsapp"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error summarizing results: {str(e)}"
#             },
#             goto="__end__"
#         )

# # WhatsApp Node
# def send_whatsapp_message(state: AgentState) -> Command[Literal["__end__"]]:
#     try:
#         payload = {
#             "messaging_product": "whatsapp",
#             "to": state['recipient_phone'],
#             "type": "text",
#             "text": {"body": state['summary']}
#         }

#         response = requests.post(
#             WHATSAPP_API_URL,
#             headers=WHATSAPP_HEADERS,
#             json=payload
#         )
#         response.raise_for_status()

#         return Command(
#             update={
#                 "whatsapp_status": {"status": "Message sent successfully", "response": response.json()},
#                 "status": "success",
#                 "error": None
#             },
#             goto="__end__"
#         )
#     except Exception as e:
#         return Command(
#             update={
#                 "status": "error",
#                 "error": f"Error sending WhatsApp message: {str(e)}"
#             },
#             goto="__end__"
#         )

# # Define the LangGraph workflow
# def create_workflow():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("searcher", search_emails)
#     workflow.add_node("summarizer", summarize_results)
#     workflow.add_node("whatsapp", send_whatsapp_message)

#     workflow.add_edge(START, "searcher")
#     workflow.add_conditional_edges(
#         "searcher",
#         lambda state: state.get("goto", "__end__"),
#         {"summarizer": "summarizer", "__end__": END}
#     )
#     workflow.add_edge("summarizer", "whatsapp")
#     workflow.add_edge("whatsapp", END)

#     return workflow.compile()

# # Initialize the graph
# graph = create_workflow()

# @app.post("/api/v1/retrieve")
# async def retrieve_info(request: LocateRequest):
#     """Endpoint to retrieve and summarize information from Gmail and send via WhatsApp"""
#     try:
#         # Initialize state
#         initial_state = AgentState(
#             query=request.query,
#             max_results=request.max_results,
#             recipient_phone=request.recipient_phone,
#             gmail_address=request.gmail_address,
#             messages=[],
#             summary="",
#             whatsapp_status={},
#             status="pending",
#             error=None
#         )

#         # Run the graph
#         final_state = graph.invoke(initial_state)

#         if final_state['status'] == "error":
#             raise HTTPException(status_code=500, detail=final_state['error'])

#         return {
#             "status": final_state['status'],
#             "results": final_state['messages'],
#             "summary": final_state['summary'],
#             "whatsapp_status": final_state['whatsapp_status']
#         }
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# workflow for web based oauth setup
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
import requests
from typing import Optional, List, TypedDict, Literal
import json
import google.generativeai as genai
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "WHATSAPP_API_KEY", "WHATSAPP_ACCESS_TOKEN", "PHONE_NUMBER_ID"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# In-memory storage for credentials (use a database in production)
CREDENTIALS_STORAGE = {}

# Pydantic model for request body
class LocateRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    recipient_phone: str
    gmail_address: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_phone
        yield cls.validate_email

    @classmethod
    def validate_phone(cls, v):
        if not re.match(r"^\d{10,15}$", v['recipient_phone']):
            raise ValueError("recipient_phone must be 10-15 digits (international format, no '+')")
        return v

    @classmethod
    def validate_email(cls, v):
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v['gmail_address']):
            raise ValueError("gmail_address must be a valid email address")
        return v

# Gmail API credential setup
def get_gmail_credentials(gmail_address: str):
    creds = CREDENTIALS_STORAGE.get(gmail_address)
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']

    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                logger.info(f"Refreshing credentials for {gmail_address}")
                creds.refresh(GoogleRequest())
                CREDENTIALS_STORAGE[gmail_address] = creds
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

    redirect_uri = f"{request.base_url}auth/callback"
    flow = get_oauth_flow(redirect_uri)
    auth_url, state = flow.authorization_url(prompt='consent')
    CREDENTIALS_STORAGE[f"{gmail_address}_state"] = state
    logger.info(f"Generated auth URL for {gmail_address}")
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
        del CREDENTIALS_STORAGE[f"{gmail_address}_state"]
        logger.info(f"Successfully stored credentials for {gmail_address}")
        return RedirectResponse(url="/auth/success")
    except Exception as e:
        logger.error(f"Failed to fetch token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch token: {str(e)}")

@app.get("/auth/success")
async def auth_success():
    return {"message": "Authentication successful. You can now use the /api/v1/retrieve endpoint."}

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise RuntimeError(f"Failed to initialize Gemini API: {str(e)}")

# WhatsApp API configuration
WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
WHATSAPP_HEADERS = {
    "Authorization": f"Bearer {os.getenv('WHATSAPP_ACCESS_TOKEN')}",
    "Content-Type": "application/json"
}

# Define the state for LangGraph
class AgentState(TypedDict):
    query: str
    max_results: int
    recipient_phone: str
    gmail_address: str
    messages: List[dict]
    summary: str
    whatsapp_status: dict
    status: str
    error: Optional[str]

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
                'date': msg_data.get('internalDate', '')
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
def summarize_results(state: AgentState) -> Command[Literal["whatsapp", "__end__"]]:
    try:
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
            goto="whatsapp"
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

# WhatsApp Node
def send_whatsapp_message(state: AgentState) -> Command[Literal["__end__"]]:
    try:
        payload = {
            "messaging_product": "whatsapp",
            "to": state['recipient_phone'],
            "type": "text",
            "text": {"body": state['summary']}
        }

        response = requests.post(
            WHATSAPP_API_URL,
            headers=WHATSAPP_HEADERS,
            json=payload
        )
        response.raise_for_status()

        logger.info(f"WhatsApp message sent to {state['recipient_phone']}")
        return Command(
            update={
                "whatsapp_status": {"status": "Message sent successfully", "response": response.json()},
                "status": "success",
                "error": None
            },
            goto="__end__"
        )
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        return Command(
            update={
                "status": "error",
                "error": f"Error sending WhatsApp message: {str(e)}"
            },
            goto="__end__"
        )

# Define the LangGraph workflow
def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("searcher", search_emails)
    workflow.add_node("summarizer", summarize_results)
    workflow.add_node("whatsapp", send_whatsapp_message)

    workflow.add_edge(START, "searcher")
    workflow.add_conditional_edges(
        "searcher",
        lambda state: state.get("goto", "__end__"),
        {"summarizer": "summarizer", "__end__": END}
    )
    workflow.add_edge("summarizer", "whatsapp")
    workflow.add_edge("whatsapp", END)

    return workflow.compile()

# Initialize the graph
graph = create_workflow()

@app.post("/api/v1/retrieve")
async def retrieve_info(request: LocateRequest):
    """Endpoint to retrieve and summarize information from Gmail and send via WhatsApp"""
    try:
        initial_state = AgentState(
            query=request.query,
            max_results=request.max_results,
            recipient_phone=request.recipient_phone,
            gmail_address=request.gmail_address,
            messages=[],
            summary="",
            whatsapp_status={},
            status="pending",
            error=None
        )

        final_state = graph.invoke(initial_state)

        if final_state['status'] == "error":
            raise HTTPException(status_code=500, detail=final_state['error'])

        logger.info(f"Successfully processed request for {request.gmail_address}")
        return {
            "status": final_state['status'],
            "results": final_state['messages'],
            "summary": final_state['summary'],
            "whatsapp_status": final_state['whatsapp_status']
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