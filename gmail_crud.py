
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv
# import logging
# from typing import Optional, Dict, Any
# from enum import Enum
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google.oauth2.credentials import Credentials
# from google.auth.transport.requests import Request as GoogleRequest
# import base64
# from email.mime.text import MIMEText
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # In-memory storage for credentials (consistent with main.py)
# CREDENTIALS_STORAGE = {}

# # Enum for CRUD operation types
# class OperationType(str, Enum):
#     CREATE = "create"
#     READ = "read"
#     UPDATE = "update"
#     DELETE = "delete"

# # Pydantic model for CRM lead (message) request
# class LeadRequest(BaseModel):
#     operation: OperationType
#     gmail_address: str
#     data: Optional[Dict[str, Any]] = None

#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate_fields

#     @classmethod
#     def validate_fields(cls, v):
#         if not v['gmail_address']:
#             raise ValueError("gmail_address is required")
#         if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v['gmail_address']):
#             raise ValueError("gmail_address must be a valid email address")
#         if v['operation'] in [OperationType.CREATE, OperationType.UPDATE]:
#             if not v['data']:
#                 raise ValueError("data is required for create and update operations")
#             if not all(key in v['data'] for key in ['to', 'subject', 'body']):
#                 raise ValueError("to, subject, and body are required for create and update operations")
#             if v['operation'] == OperationType.UPDATE and 'id' not in v['data']:
#                 raise ValueError("id is required for update operation")
#         if v['operation'] == OperationType.DELETE and (not v['data'] or 'ids' not in v['data']):
#             raise ValueError("data with 'ids' is required for delete operation")
#         return v

# # Gmail API credential setup (from main.py)
# def get_gmail_credentials(gmail_address: str):
#     creds = CREDENTIALS_STORAGE.get(gmail_address)
#     scopes = ['https://www.googleapis.com/auth/gmail.modify']

#     # Try loading from file if not in memory
#     if not creds:
#         safe_email = gmail_address.replace('@', '_').replace('.', '_')
#         token_file = f"token_{safe_email}.json"
#         if os.path.exists(token_file):
#             try:
#                 creds = Credentials.from_authorized_user_file(token_file, scopes)
#                 CREDENTIALS_STORAGE[gmail_address] = creds
#                 logger.info(f"Loaded credentials from {token_file} for {gmail_address}")
#             except Exception as e:
#                 logger.error(f"Failed to load credentials from {token_file}: {str(e)}")
#                 creds = None

#     if creds and not creds.valid:
#         if creds.expired and creds.refresh_token:
#             try:
#                 logger.info(f"Refreshing credentials for {gmail_address}")
#                 creds.refresh(GoogleRequest())
#                 CREDENTIALS_STORAGE[gmail_address] = creds
#                 safe_email = gmail_address.replace('@', '_').replace('.', '_')
#                 token_file = f"token_{safe_email}.json"
#                 with open(token_file, 'w') as token:
#                     token.write(creds.to_json())
#                 logger.info(f"Saved refreshed credentials to {token_file}")
#             except Exception as e:
#                 logger.error(f"Failed to refresh credentials: {str(e)}")
#                 raise RuntimeError(f"Failed to refresh credentials: {str(e)}")
#         else:
#             creds = None

#     if not creds:
#         raise HTTPException(
#             status_code=401,
#             detail=f"No valid credentials for {gmail_address}. Please authenticate via /auth/start."
#         )

#     return creds

# # Endpoint for CRUD operations on Gmail messages
# @app.post("/api/v1/crm/leads")
# async def manage_leads(request: LeadRequest):
#     try:
#         # Authenticate Gmail
#         creds = get_gmail_credentials(request.gmail_address)
#         service = build('gmail', 'v1', credentials=creds, cache_discovery=False)

#         # Handle CRUD operations
#         if request.operation == OperationType.CREATE:
#             if not request.data or 'to' not in request.data or 'subject' not in request.data or 'body' not in request.data:
#                 raise HTTPException(status_code=400, detail="to, subject, and body are required for create operation")
            
#             # Create a message
#             message = MIMEText(request.data['body'])
#             message['to'] = request.data['to']
#             message['subject'] = request.data['subject']
#             raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
#             # Create draft
#             draft = service.users().drafts().create(
#                 userId='me',
#                 body={'message': {'raw': raw_message}}
#             ).execute()
            
#             # Send the draft
#             sent_message = service.users().drafts().send(userId='me', body={'id': draft['id']}).execute()
#             logger.info(f"Created and sent message with ID {sent_message['id']} for {request.gmail_address}")
#             return {
#                 "status": "success",
#                 "operation": request.operation,
#                 "result": sent_message,
#                 "message": f"Message created and sent with ID {sent_message['id']}"
#             }

#         elif request.operation == OperationType.READ:
#             query = ''
#             if request.data and 'ids' in request.data:
#                 results = []
#                 for msg_id in request.data['ids']:
#                     message = service.users().messages().get(userId='me', id=msg_id).execute()
#                     results.append(message)
#             elif request.data and 'search_term' in request.data:
#                 query = request.data['search_term']
#                 results = service.users().messages().list(
#                     userId='me',
#                     q=query,
#                     maxResults=request.data.get('limit', 10)
#                 ).execute().get('messages', [])
#                 if results:
#                     results = [service.users().messages().get(userId='me', id=msg['id']).execute() for msg in results]
#             else:
#                 results = service.users().messages().list(
#                     userId='me',
#                     maxResults=request.data.get('limit', 10) if request.data else 10
#                 ).execute().get('messages', [])
#                 if results:
#                     results = [service.users().messages().get(userId='me', id=msg['id']).execute() for msg in results]
            
#             logger.info(f"Retrieved {len(results)} messages for {request.gmail_address}")
#             return {
#                 "status": "success",
#                 "operation": request.operation,
#                 "result": results,
#                 "message": f"Retrieved {len(results)} messages"
#             }

#         elif request.operation == OperationType.UPDATE:
#             if not request.data or 'id' not in request.data or 'to' not in request.data or 'subject' not in request.data or 'body' not in request.data:
#                 raise HTTPException(status_code=400, detail="id, to, subject, and body are required for update operation")
            
#             message_id = request.data['id']
            
#             # Create a new message with updated content
#             message = MIMEText(request.data['body'])
#             message['to'] = request.data['to']
#             message['subject'] = request.data['subject']
#             raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
#             # Create a new draft
#             draft = service.users().drafts().create(
#                 userId='me',
#                 body={'message': {'raw': raw_message}}
#             ).execute()
            
#             # Send the new draft
#             updated_message = service.users().drafts().send(userId='me', body={'id': draft['id']}).execute()
            
#             # Optionally delete the original message
#             if request.data.get('delete_original', False):
#                 try:
#                     service.users().messages().delete(userId='me', id=message_id).execute()
#                     logger.info(f"Deleted original message with ID {message_id} for {request.gmail_address}")
#                 except HttpError as e:
#                     logger.warning(f"Failed to delete original message {message_id}: {str(e)}")
            
#             logger.info(f"Created and sent updated message with ID {updated_message['id']} for {request.gmail_address}")
#             return {
#                 "status": "success",
#                 "operation": request.operation,
#                 "result": updated_message,
#                 "message": f"Updated message created and sent, new ID: {updated_message['id']}"
#             }

#         elif request.operation == OperationType.DELETE:
#             if not request.data or 'ids' not in request.data:
#                 raise HTTPException(status_code=400, detail="ids are required for delete operation")
            
#             message_ids = request.data['ids']
#             for msg_id in message_ids:
#                 service.users().messages().delete(userId='me', id=msg_id).execute()
#             logger.info(f"Deleted messages with IDs: {message_ids} for {request.gmail_address}")
#             return {
#                 "status": "success",
#                 "operation": request.operation,
#                 "result": [],
#                 "message": f"Messages with IDs {message_ids} deleted"
#             }

#     except HttpError as e:
#         logger.error(f"Gmail API error for {request.gmail_address}: {str(e)}")
#         status_code = e.resp.status
#         raise HTTPException(status_code=status_code, detail=f"Gmail API error: {str(e)}")
#     except HTTPException as e:
#         logger.error(f"HTTP error for {request.gmail_address}: {str(e)}")
#         raise e
#     except Exception as e:
#         logger.error(f"Unexpected error for {request.gmail_address}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# Added Read with Gmail links for CRUD  
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any
from enum import Enum
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
import base64
from email.mime.text import MIMEText
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# In-memory storage for credentials
CREDENTIALS_STORAGE = {}

# Enum for CRUD operation types
class OperationType(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

# Pydantic model for CRM lead (message) request
class LeadRequest(BaseModel):
    operation: OperationType
    gmail_address: str
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields

    @classmethod
    def validate_fields(cls, v):
        if not v['gmail_address']:
            raise ValueError("gmail_address is required")
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v['gmail_address']):
            raise ValueError("gmail_address must be a valid email address")
        if v['operation'] in [OperationType.CREATE, OperationType.UPDATE]:
            if not v['data']:
                raise ValueError("data is required for create and update operations")
            if not all(key in v['data'] for key in ['to', 'subject', 'body']):
                raise ValueError("to, subject, and body are required for create and update operations")
            if v['operation'] == OperationType.UPDATE and 'id' not in v['data']:
                raise ValueError("id is required for update operation")
        if v['operation'] == OperationType.DELETE and (not v['data'] or 'ids' not in v['data']):
            raise ValueError("data with 'ids' is required for delete operation")
        return v

# Gmail API credential setup
def get_gmail_credentials(gmail_address: str):
    creds = CREDENTIALS_STORAGE.get(gmail_address)
    scopes = ['https://www.googleapis.com/auth/gmail.modify']

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

# Endpoint for CRUD operations on Gmail messages
@app.post("/api/v1/crm/leads")
async def manage_leads(request: LeadRequest):
    try:
        # Authenticate Gmail
        creds = get_gmail_credentials(request.gmail_address)
        service = build('gmail', 'v1', credentials=creds, cache_discovery=False)

        # Handle CRUD operations
        if request.operation == OperationType.CREATE:
            if not request.data or 'to' not in request.data or 'subject' not in request.data or 'body' not in request.data:
                raise HTTPException(status_code=400, detail="to, subject, and body are required for create operation")
            
            # Create a message
            message = MIMEText(request.data['body'])
            message['to'] = request.data['to']
            message['subject'] = request.data['subject']
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Create draft
            draft = service.users().drafts().create(
                userId='me',
                body={'message': {'raw': raw_message}}
            ).execute()
            
            # Send the draft
            sent_message = service.users().drafts().send(userId='me', body={'id': draft['id']}).execute()
            logger.info(f"Created and sent message with ID {sent_message['id']} for {request.gmail_address}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": sent_message,
                "message": f"Message created and sent with ID {sent_message['id']}"
            }

        elif request.operation == OperationType.READ:
            results = []
            if request.data and 'ids' in request.data:
                for msg_id in request.data['ids']:
                    msg_data = service.users().messages().get(userId='me', id=msg_id).execute()
                    headers = msg_data.get('payload', {}).get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
                    results.append({
                        'id': msg_data['id'],
                        'snippet': msg_data.get('snippet', ''),
                        'from': sender,
                        'subject': subject,
                        'date': msg_data.get('internalDate', ''),
                        'link': f"https://mail.google.com/mail/u/0/#inbox/{msg_data['id']}"
                    })
            elif request.data and 'search_term' in request.data:
                query = request.data['search_term']
                messages = service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=request.data.get('limit', 10)
                ).execute().get('messages', [])
                for msg in messages:
                    msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                    headers = msg_data.get('payload', {}).get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
                    results.append({
                        'id': msg_data['id'],
                        'snippet': msg_data.get('snippet', ''),
                        'from': sender,
                        'subject': subject,
                        'date': msg_data.get('internalDate', ''),
                        'link': f"https://mail.google.com/mail/u/0/#inbox/{msg_data['id']}"
                    })
            else:
                messages = service.users().messages().list(
                    userId='me',
                    maxResults=request.data.get('limit', 10) if request.data else 10
                ).execute().get('messages', [])
                for msg in messages:
                    msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                    headers = msg_data.get('payload', {}).get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
                    results.append({
                        'id': msg_data['id'],
                        'snippet': msg_data.get('snippet', ''),
                        'from': sender,
                        'subject': subject,
                        'date': msg_data.get('internalDate', ''),
                        'link': f"https://mail.google.com/mail/u/0/#inbox/{msg_data['id']}"
                    })
            
            logger.info(f"Retrieved {len(results)} messages for {request.gmail_address}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": results,
                "message": f"Retrieved {len(results)} messages"
            }

        elif request.operation == OperationType.UPDATE:
            if not request.data or 'id' not in request.data or 'to' not in request.data or 'subject' not in request.data or 'body' not in request.data:
                raise HTTPException(status_code=400, detail="id, to, subject, and body are required for update operation")
            
            message_id = request.data['id']
            
            # Create a new message with updated content
            message = MIMEText(request.data['body'])
            message['to'] = request.data['to']
            message['subject'] = request.data['subject']
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Create a new draft
            draft = service.users().drafts().create(
                userId='me',
                body={'message': {'raw': raw_message}}
            ).execute()
            
            # Send the new draft
            updated_message = service.users().drafts().send(userId='me', body={'id': draft['id']}).execute()
            
            # Optionally delete the original message
            if request.data.get('delete_original', False):
                try:
                    service.users().messages().delete(userId='me', id=message_id).execute()
                    logger.info(f"Deleted original message with ID {message_id} for {request.gmail_address}")
                except HttpError as e:
                    logger.warning(f"Failed to delete original message {message_id}: {str(e)}")
            
            logger.info(f"Created and sent updated message with ID {updated_message['id']} for {request.gmail_address}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": updated_message,
                "message": f"Updated message created and sent, new ID: {updated_message['id']}"
            }

        elif request.operation == OperationType.DELETE:
            if not request.data or 'ids' not in request.data:
                raise HTTPException(status_code=400, detail="ids are required for delete operation")
            
            message_ids = request.data['ids']
            for msg_id in message_ids:
                service.users().messages().delete(userId='me', id=msg_id).execute()
            logger.info(f"Deleted messages with IDs: {message_ids} for {request.gmail_address}")
            return {
                "status": "success",
                "operation": request.operation,
                "result": [],
                "message": f"Messages with IDs {message_ids} deleted"
            }

    except HttpError as e:
        logger.error(f"Gmail API error for {request.gmail_address}: {str(e)}")
        status_code = e.resp.status
        raise HTTPException(status_code=status_code, detail=f"Gmail API error: {str(e)}")
    except HTTPException as e:
        logger.error(f"HTTP error for {request.gmail_address}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error for {request.gmail_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)