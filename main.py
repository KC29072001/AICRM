from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import requests
from typing import Optional, List, TypedDict, Literal
import json
import google.generativeai as genai
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Load environment variables
load_dotenv()

app = FastAPI()

# Pydantic model for request body
class LocateRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    recipient_phone: str

# Gmail API credential setup
def get_gmail_credentials():
    creds = None
    token_path = "token.json"
    credentials_path = "credentials.json"
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path,
                scopes
            )
            creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

    return creds

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

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
    messages: List[dict]
    summary: str
    whatsapp_status: dict
    status: str
    error: Optional[str]


# Email Search Node
def search_emails(state: AgentState) -> Command[Literal["summarizer", "__end__"]]:
    try:
        creds = get_gmail_credentials()
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', q=state['query'], maxResults=state['max_results']).execute()
        messages = results.get('messages', [])
        message_details = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            message_details.append({
                'id': msg['id'],
                'snippet': msg_data['snippet'],
                'from': sender,
                'subject': subject,
                'date': msg_data.get('internalDate')
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
                "status": "success"
            },
            goto="summarizer"
        )
    except Exception as e:
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
                "status": "success"
            },
            goto="whatsapp"
        )
    except Exception as e:
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
        
        return Command(
            update={
                "whatsapp_status": {"status": "Message sent successfully"},
                "status": "success"
            },
            goto="__end__"
        )
    except Exception as e:
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
    workflow.add_edge("searcher", "summarizer")
    workflow.add_edge("summarizer", "whatsapp")
    
    return workflow.compile()

# Initialize the graph
graph = create_workflow()

@app.post("/api/v1/retrieve")
async def retrieve_info(request: LocateRequest):
    """Endpoint to retrieve and summarize information from Gmail and send via WhatsApp"""
    try:
        # Initialize state
        initial_state = AgentState(
            query=request.query,
            max_results=request.max_results,
            recipient_phone=request.recipient_phone,
            messages=[],
            summary="",
            whatsapp_status={},
            status="pending",
            error=None
        )
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        if final_state['status'] == "error":
            raise HTTPException(status_code=500, detail=final_state['error'])
        
        return {
            "status": final_state['status'],
            "results": final_state['messages'],
            "summary": final_state['summary'],
            "whatsapp_status": final_state['whatsapp_status']
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

