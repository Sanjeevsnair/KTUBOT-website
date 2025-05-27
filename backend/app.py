from fastapi import Depends, FastAPI, HTTPException, Request, exceptions, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import firebase_admin
from firebase_admin import auth, credentials
from pydantic import BaseModel
import json
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai
import os
import logging
import hashlib
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
from io import BytesIO
import requests
from urllib3 import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="EduChat AI Backend", version="1.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load service account credentials
SERVICE_ACCOUNT_FILE = "ktubot-7a64f8f9fac7.json"
GOOGLE_DRIVE_FOLDER_ID = "15gnvPIxP4oqFghT1f-3lyciYApL7Qget"
credentialss = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# Initialize Drive service
drive_service = build(
    "drive",
    "v3",
    credentials=credentialss,
    retry=Retry(deadline=30),
    static_discovery=False,
)

import re

# Initialize Firebase Admin
cred = credentials.Certificate(
    "ktubot-c3495-firebase-adminsdk-fbsvc-5ec25bbfae.json"
)  # Download from Firebase Console
firebase_admin.initialize_app(cred)

# Security
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        if not credentials.scheme == "Bearer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authentication scheme",
            )

        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except exceptions.FirebaseError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


# Add these models
class FirebaseLoginRequest(BaseModel):
    id_token: str


class EmailPasswordLoginRequest(BaseModel):
    email: str
    password: str


class EmailPasswordRegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


# Add these endpoints to your FastAPI app
@app.post("/api/auth/firebase-login")
async def firebase_login(request: FirebaseLoginRequest):
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(request.id_token)
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/email-register")
async def email_register(user_data: EmailPasswordRegisterRequest):
    try:
        user = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name,
        )
        return {"uid": user.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/email-login")
async def email_login(login_data: EmailPasswordLoginRequest):
    try:
        # This endpoint is just for your backend to verify, frontend will handle actual sign-in
        user = auth.get_user_by_email(login_data.email)
        return {"uid": user.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/session")
async def create_session_token(user: dict = Depends(get_current_user)):
    """Create a session token for the authenticated user"""
    try:
        # Create a custom token that can be used on the client side
        custom_token = auth.create_custom_token(user["uid"])
        return {"token": custom_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/verify")
async def verify_token(token: str):
    """Verify a Firebase ID token"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Example protected endpoint
@app.get("/api/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    return {"message": f"Hello {user.get('email')}, you're authenticated!"}


# Example: List files in a folder
def list_files(folder_id):
    results = (
        drive_service.files()
        .list(q=f"'{folder_id}' in parents", fields="files(id, name)")
        .execute()
    )
    return results.get("files", [])


# Configure Gemini
genai.configure(api_key="")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# Pydantic models for request/response validation
class DepartmentRequest(BaseModel):
    department: str


class SemesterRequest(BaseModel):
    department: str
    semester: str


class SubjectRequest(BaseModel):
    department: str
    semester: str
    subject: str


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    isFormatted: Optional[bool] = None
    containsImages: Optional[bool] = None


class ChatRequest(BaseModel):
    department: str
    semester: str
    subject: str
    question: str
    chat_history: Optional[List[ChatMessage]] = []


# Cache for subject data
subject_cache: Dict[str, Dict] = {}
chat_contexts: Dict[str, List[Dict[str, str]]] = {}


def get_context_key(department: str, semester: str, subject: str) -> str:
    """Generate a unique key for chat context"""
    return f"{department}_{semester}_{subject}"


# Update the generate_prompt function in your FastAPI backend
def generate_prompt(
    subject_data: Dict, question: str, chat_history: List[Dict[str, str]]
) -> str:
    """Generate prompt with context-aware diagram instructions and LaTeX to plain text conversion"""
    syllabus_content = []
    diagram_map = {}  # Track diagrams with their context

    for module in subject_data["content"]["modules"]:
        for topic in module["topics"]:
            # Store topic context and diagrams
            topic_context = f"Module {module['module_number']}: {module['module_title']} - {topic['topic_title']}"
            if topic["content"]["diagrams"]:
                for diagram in topic["content"]["diagrams"]:
                    diagram_map[diagram] = {
                        "context": topic_context,
                        "content": latex_to_plaintext(topic["content"]["text"]),
                    }

            syllabus_content.append(f"## {topic_context}")
            syllabus_content.append(
                latex_to_plaintext(topic["content"]["text"].strip())
            )

            for subtopic in topic.get("subtopics", []):
                # Store subtopic context and diagrams
                subtopic_context = f"{topic_context} > {subtopic['subtopic_title']}"
                if subtopic["content"]["diagrams"]:
                    for diagram in subtopic["content"]["diagrams"]:
                        diagram_map[diagram] = {
                            "context": subtopic_context,
                            "content": latex_to_plaintext(subtopic["content"]["text"]),
                        }

                syllabus_content.append(f"### {subtopic['subtopic_title']}")
                syllabus_content.append(
                    latex_to_plaintext(subtopic["content"]["text"].strip())
                )

    syllabus_text = "\n".join(syllabus_content)

    # Prepare chat history context with LaTeX conversion
    history_context = []
    for msg in chat_history[-3:]:
        role = "Student" if msg["role"] == "user" else "You"
        history_context.append(f"{role}: {latex_to_plaintext(msg['content'])}")

    history_text = (
        "\n\nPrevious discussion:\n" + "\n".join(history_context)
        if history_context
        else ""
    )

    diagram_instructions = []
    for diagram, info in diagram_map.items():
        diagram_instructions.append(
            f"- Diagram URL: {diagram}\n"
            f"  Context: {info['context']}\n"
            f"  Relevant content: {info['content'][:200]}..."
        )

    # Convert LaTeX in the question
    processed_question = latex_to_plaintext(question)

    # Construct the strict prompt
    # Construct the strict prompt (continued)
    prompt = f"""
**Syllabus Content**:
{syllabus_text}
{history_text}

**Current Question**:
{latex_to_plaintext(question)}

**Available Diagrams with Context**:
{"\n".join(diagram_instructions) if diagram_instructions else "No diagrams available"}

Your response MUST use this exact format:

    ## [Main Topic Title]  <!-- H2 for main sections -->
    ### [Subheading]      <!-- H3 for subsections -->
    [Paragraph text]      <!-- Regular paragraphs -->
    • Bullet point 1      <!-- For lists -->
    • Bullet point 2
    **Bold terms**        <!-- For emphasis -->
    `code terms`          <!-- For technical terms -->
    
**Diagram Inclusion Rules**:
1. Analyze which diagrams are MOST relevant to the specific question
2. Include ONLY diagrams that directly illustrate concepts mentioned in the question
3. Use this exact format for each diagram: 
   [DIAGRAM: full_url "Accurate caption explaining relevance to question"]
4. Place each diagram IMMEDIATELY after the text it illustrates
5. Never include diagrams that aren't directly relevant

**Response Requirements**:
1. Answer must directly address the question using ONLY the syllabus content above
2. If the question is out of syllabus, respond ONLY with: "This topic is not covered in the syllabus."
3. Format requirements:
   - Use proper markdown formatting as shown above
   - ALWAYS use bullet points (•)
   - Each point should be concise (max 1 sentence)
   - Skip ALL introductions/conclusions
   - NEVER say "according to the syllabus" or similar
   - For each response, strictly follow this formatting rule for the title:
     - Display the Module Number and Module Name as the main heading
4. Content priority:
   - First match subtopic content if relevant
   - Then match topic content
   - Finally module overview if needed
5. STRICTLY convert any LaTeX math expressions to plain text:
   - $E=mc^2$ → E=mc^2
   - \frac{'a'}{'b'} → a/b
   - \sqrt{'x'} → sqrt(x)
   - Remove all other LaTeX commands

**Response**:
"""
    return prompt


def latex_to_plaintext(text: str) -> str:
    """Convert LaTeX math blocks to plain text"""
    # Handle inline math $...$
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    # Handle display math \[...\]
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text)
    # Handle other common LaTeX math environments
    text = re.sub(r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}", r"\1", text)
    text = re.sub(r"\\begin\{align\*\}(.*?)\\end\{align\*\}", r"\1", text)
    # Remove other LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    return text


def list_drive_folder(folder_id):
    """List all files in a Google Drive folder"""
    try:
        results = (
            drive_service.files()
            .list(q=f"'{folder_id}' in parents", fields="files(id, name, mimeType)")
            .execute()
        )
        return results.get("files", [])
    except Exception as e:
        logger.error(f"Error listing Drive folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing Google Drive")


def get_file_content(file_id):
    """Get content of a file from Google Drive"""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_content = request.execute()
        return file_content
    except Exception as e:
        logger.error(f"Error getting file content: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error fetching file from Google Drive"
        )


def load_subject_data_from_drive(department: str, semester: str, subject: str) -> Dict:
    """Load subject data from Google Drive with caching"""
    cache_key = f"{department}_{semester}_{subject}"

    if cache_key in subject_cache:
        return subject_cache[cache_key]

    try:
        # First find the department folder
        dept_folders = list_drive_folder(GOOGLE_DRIVE_FOLDER_ID)
        dept_folder = next(
            (
                f
                for f in dept_folders
                if f["name"] == department
                and f["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )

        if not dept_folder:
            raise HTTPException(status_code=404, detail="Department not found")

        # Then find the semester folder
        semester_folders = list_drive_folder(dept_folder["id"])
        semester_folder = next(
            (
                f
                for f in semester_folders
                if f["name"] == semester
                and f["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )

        if not semester_folder:
            raise HTTPException(status_code=404, detail="Semester not found")

        # Then find the subject file
        subject_files = list_drive_folder(semester_folder["id"])
        subject_file = next(
            (f for f in subject_files if f["name"] == f"{subject}.json"), None
        )

        if not subject_file:
            raise HTTPException(status_code=404, detail="Subject not found")

        # Get the file content
        file_content = get_file_content(subject_file["id"])
        data = json.loads(file_content.decode("utf-8"))
        subject_cache[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"Error loading subject data from Drive: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading subject data")


# Update your existing load_subject_data function to use the Drive version
def load_subject_data(department: str, semester: str, subject: str) -> Dict:
    return load_subject_data_from_drive(department, semester, subject)


# Alternative approach if using direct public links (without API)
def load_subject_data_direct(department: str, semester: str, subject: str) -> Dict:
    """Alternative method if files are directly accessible via public URLs"""
    cache_key = f"{department}_{semester}_{subject}"

    if cache_key in subject_cache:
        return subject_cache[cache_key]

    try:
        # Construct the public URL based on your folder structure
        # This depends on how you've shared your Google Drive files
        file_url = f"https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

        response = requests.get(file_url)
        response.raise_for_status()

        data = response.json()
        subject_cache[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"Error loading subject data directly: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading subject data")


# API Endpoints (remain largely the same, but will use Drive functions)
@app.get("/api/departments", response_model=List[str])
async def get_departments():
    """Get list of available departments from Google Drive"""
    try:
        folders = list_drive_folder(GOOGLE_DRIVE_FOLDER_ID)
        departments = [
            f["name"]
            for f in folders
            if f["mimeType"] == "application/vnd.google-apps.folder"
        ]
        return departments
    except Exception as e:
        logger.error(f"Error getting departments: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching departments")


@app.post("/api/semesters", response_model=List[str])
async def get_semesters(request: DepartmentRequest):
    """Get list of semesters for a department from Google Drive"""
    try:
        # First find the department folder
        dept_folders = list_drive_folder(GOOGLE_DRIVE_FOLDER_ID)
        dept_folder = next(
            (
                f
                for f in dept_folders
                if f["name"] == request.department
                and f["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )

        if not dept_folder:
            raise HTTPException(status_code=404, detail="Department not found")

        # Get semester folders
        semester_folders = list_drive_folder(dept_folder["id"])
        semesters = [
            f["name"]
            for f in semester_folders
            if f["mimeType"] == "application/vnd.google-apps.folder"
        ]
        return semesters
    except Exception as e:
        logger.error(f"Error getting semesters: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching semesters")


@app.post("/api/subjects", response_model=List[str])
async def get_subjects(request: SemesterRequest):
    """Get list of subjects for a semester from Google Drive"""
    try:
        # First find the department folder
        dept_folders = list_drive_folder(GOOGLE_DRIVE_FOLDER_ID)
        dept_folder = next(
            (
                f
                for f in dept_folders
                if f["name"] == request.department
                and f["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )

        if not dept_folder:
            raise HTTPException(status_code=404, detail="Department not found")

        # Then find the semester folder
        semester_folders = list_drive_folder(dept_folder["id"])
        semester_folder = next(
            (
                f
                for f in semester_folders
                if f["name"] == request.semester
                and f["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )

        if not semester_folder:
            raise HTTPException(status_code=404, detail="Semester not found")

        # Get subject files
        subject_files = list_drive_folder(semester_folder["id"])
        subjects = [
            f["name"].replace(".json", "")
            for f in subject_files
            if f["name"].endswith(".json")
        ]
        return subjects
    except Exception as e:
        logger.error(f"Error getting subjects: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching subjects")


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """Handle chat interactions with context-aware responses"""
    try:
        # Clean the chat history (convert HTML to text if needed)
        cleaned_history = []
        for msg in request.chat_history:
            content = msg.content
            if msg.isFormatted:
                # Convert HTML to plain text if needed
                content = re.sub("<[^<]+?>", "", content)  # Simple HTML tag removal
            cleaned_history.append({"role": msg.role, "content": content})

        # Load subject data
        subject_data = load_subject_data(
            request.department, request.semester, request.subject
        )

        # Generate context-aware prompt using cleaned history
        prompt = generate_prompt(subject_data, request.question, cleaned_history)

        logger.info(
            f"Generated prompt: {prompt[:500]}..."
        )  # Log first 500 chars of prompt

        # Get response from Gemini
        response = gemini_model.generate_content(prompt)

        # Get the text response
        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from AI")

        # Update chat context
        context_key = get_context_key(
            request.department, request.semester, request.subject
        )
        if context_key not in chat_contexts:
            chat_contexts[context_key] = []

        # Add to chat history (both question and response)
        chat_contexts[context_key].extend(
            [
                {"role": "user", "content": request.question},
                {"role": "assistant", "content": response.text},
            ]
        )

        # Limit chat history to last 10 messages
        chat_contexts[context_key] = chat_contexts[context_key][-10:]

        return {
            "answer": response.text,
            "sources": {
                "subject": request.subject,
                "department": request.department,
                "semester": request.semester,
            },
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend if needed
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("frontend/index.html")


# Mount static files if needed

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
