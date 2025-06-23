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
GOOGLE_DRIVE_FOLDER_ID = "15gnvPIxP4oqFghT1f-3lyciYApL7Qget"
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv()

# Parse credentials from environment variable
creds_json = os.environ.get("GOOGLE_CREDS")
if not creds_json:
    raise ValueError("GOOGLE_CREDS environment variable not set")

creds_dict = json.loads(creds_json)
credentialss = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# Initialize Drive service
drive_service = build(
    "drive",
    "v3",
    credentials=credentialss,
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


def extract_and_convert_latex_tables(text: str) -> tuple[str, Dict[str, str]]:
    """Extract LaTeX tables and convert them to HTML tables with placeholders"""
    table_pattern = re.compile(
        r'\\begin\{table\}.*?\\end\{table\}',
        re.DOTALL | re.IGNORECASE
    )
    
    tables = {}
    table_counter = 0
    
    def replace_table(match):
        nonlocal table_counter
        table_latex = match.group(0)
        table_id = f"TABLE_PLACEHOLDER_{table_counter}"
        table_counter += 1
        
        # Convert LaTeX table to HTML
        html_table = latex_table_to_html(table_latex)
        tables[table_id] = html_table
        
        return f"[{table_id}]"
    
    # Replace tables with placeholders
    text_with_placeholders = table_pattern.sub(replace_table, text)
    
    return text_with_placeholders, tables


def latex_table_to_html(latex_table: str) -> str:
    """Convert LaTeX table to styled HTML table - preserves all data, removes only formatting"""
    try:
        # Extract table content between \begin{tabular} and \end{tabular}
        tabular_pattern = re.compile(
            r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
            re.DOTALL
        )
        
        tabular_match = tabular_pattern.search(latex_table)
        if not tabular_match:
            return "<p>Table formatting error</p>"
        
        column_spec = tabular_match.group(1).strip()  # Extract column specification
        table_content = tabular_match.group(2).strip()
        
        # Extract caption if present
        caption_pattern = re.compile(r'\\caption\{([^}]*)\}')
        caption_match = caption_pattern.search(latex_table)
        caption = caption_match.group(1) if caption_match else ""
        
        # STEP 1: Remove LaTeX formatting commands (but preserve content)
        table_content = re.sub(r'\\rowcolor\{[^}]*\}', '', table_content)
        table_content = re.sub(r'\\midrule', '', table_content)
        table_content = re.sub(r'\\bottomrule', '', table_content)
        table_content = re.sub(r'\\toprule', '', table_content)
        table_content = re.sub(r'\\arrayrulecolor\{[^}]*\}', '', table_content)
        table_content = re.sub(r'\\columncolor\{[^}]*\}', '', table_content)
        table_content = re.sub(r'\\cellcolor\{[^}]*\}', '', table_content)
        
        # STEP 2: Split into rows first, then clean each row
        # Split by \\ (end of row marker)
        raw_rows = re.split(r'\\\\', table_content)
        
        # Process each row
        rows = []
        for i, raw_row in enumerate(raw_rows):
            raw_row = raw_row.strip()
            
            # Skip completely empty rows
            if not raw_row:
                continue
            
            # CRITICAL: Only remove if the row contains ONLY column specification
            # Check if this row is purely a column specification that leaked in
            # This is more precise - only remove if it matches exactly the column spec pattern
            if re.match(r'^[lcr\s>{}|]*$', raw_row) and len(raw_row) < 50:
                # Additional check: see if it matches the actual column specification
                clean_row = re.sub(r'[^lcr>{}]', '', raw_row)
                clean_spec = re.sub(r'[^lcr>{}]', '', column_spec)
                if clean_row == clean_spec or not any(c.isalnum() for c in raw_row):
                    continue
            
            # Split by & (column separator)
            cells = re.split(r'(?<!\\)&', raw_row)
            cleaned_cells = []
            
            for j, cell in enumerate(cells):
                cell = cell.strip()
                
                # PRESERVE CONTENT: Only clean formatting, don't remove content
                # Check if this cell is ONLY a column specification fragment
                if j == 0 and i == 0 and re.match(r'^[lcr\s>{}]*$', cell) and len(cell) < 20:
                    # This might be a leaked column spec in first cell of first row
                    # But only remove if it exactly matches part of the column specification
                    clean_cell = re.sub(r'[^lcr>{}]', '', cell)
                    clean_spec = re.sub(r'[^lcr>{}]', '', column_spec)
                    if clean_cell in clean_spec and not any(c.isalnum() for c in cell):
                        cell = ''  # Remove only if it's clearly a column spec leak
                
                # Clean LaTeX commands but preserve all actual content
                cell = re.sub(r'\\textsubscript\{([^}]*)\}', r'<sub>\1</sub>', cell)
                cell = re.sub(r'\\textsuperscript\{([^}]*)\}', r'<sup>\1</sup>', cell)
                cell = re.sub(r'\\textsuper\{([^}]*)\}', r'<sup>\1</sup>', cell)
                cell = re.sub(r'\\rightarrow', '→', cell)
                cell = re.sub(r'\\leftarrow', '←', cell)
                cell = re.sub(r'\\newline', '<br>', cell)
                cell = re.sub(r'\$([^$]*)\$', r'\1', cell)  # Remove math mode delimiters
                cell = re.sub(r'\\textbf\{([^}]*)\}', r'<strong>\1</strong>', cell)
                cell = re.sub(r'\\textit\{([^}]*)\}', r'<em>\1</em>', cell)
                
                # Remove LaTeX commands but preserve their content
                cell = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', cell)  # Extract content from commands
                cell = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?', '', cell)  # Remove standalone commands
                
                # Clean up remaining LaTeX syntax
                cell = re.sub(r'[{}]', '', cell)
                cell = cell.strip()
                
                cleaned_cells.append(cell)
            
            # Add all rows that have any content (even if some cells are empty)
            if cleaned_cells:
                # Ensure we don't lose rows just because they have empty cells
                rows.append(cleaned_cells)
        
        if not rows:
            return "<p>No table data found</p>"
        
        # STEP 3: Normalize column count - ensure all rows have same number of columns
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append('')  # Pad with empty cells
        
        # Generate HTML table
        html_parts = []
        html_parts.append('<div class="table-container">')
        html_parts.append('<table class="latex-table">')
        
        # Determine if first row should be header
        # Simple heuristic: if first row has different formatting or all non-empty cells
        has_header = True
        if len(rows) > 1:
            first_row_content = sum(1 for cell in rows[0] if cell.strip())
            if first_row_content == 0:
                has_header = False
        
        if has_header and rows:
            # Add header row
            html_parts.append('<thead>')
            html_parts.append('<tr class="header-row">')
            for cell in rows[0]:
                html_parts.append(f'<th>{cell if cell else "&nbsp;"}</th>')
            html_parts.append('</tr>')
            html_parts.append('</thead>')
            
            # Add body rows (skip first row since it's the header)
            html_parts.append('<tbody>')
            for row in rows[1:]:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{cell if cell else "&nbsp;"}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody>')
        else:
            # No header, all rows are data
            html_parts.append('<tbody>')
            for row in rows:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{cell if cell else "&nbsp;"}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody>')
        
        html_parts.append('</table>')
       
        html_parts.append('</div>')
        
        # Add CSS styling
        css_style = """
        <style>
        .table-container {
            margin: 20px 0;
            overflow-x: auto;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        .latex-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: rgba(30, 41, 59, 0.7);
            margin: 0;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .latex-table thead th {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%);
            color: #ffffff;
            font-weight: 600;
            padding: 16px 20px;
            text-align: left;
            border: none;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
        }
        
        .latex-table thead th::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: rgba(255, 255, 255, 0.1);
        }
        
        .latex-table tbody tr {
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            transition: background-color 0.2s ease;
        }
        
        .latex-table tbody tr:nth-child(even) {
            background: rgba(51, 65, 85, 0.3);
        }
        
        .latex-table tbody tr:hover {
            background: rgba(99, 102, 241, 0.08);
        }
        
        .latex-table td {
            padding: 14px 20px;
            border: none;
            font-size: 14px;
            color: #94a3b8;
            vertical-align: top;
            line-height: 1.5;
        }
        
        .latex-table td:first-child {
            font-weight: 500;
            color: #f8fafc;
            position: relative;
        }
        
        .latex-table tbody tr:hover td:first-child::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: #6366f1;
            border-radius: 0 2px 2px 0;
        }
        
        .table-caption {
            text-align: center;
            font-weight: 500;
            color: #e2e8f0;
            margin-top: 12px;
            font-size: 13px;
            font-style: italic;
        }
        
        .latex-table sub {
            font-size: 75%;
            line-height: 0;
            position: relative;
            vertical-align: baseline;
            bottom: -0.25em;
        }
        
        .latex-table sup {
            font-size: 75%;
            line-height: 0;
            position: relative;
            vertical-align: baseline;
            top: -0.5em;
        }
        
        .latex-table th, .latex-table td {
            min-width: 80px;
        }
        
        /* Subtle row highlighting */
        .latex-table tbody tr:hover td {
            color: #cbd5e1;
        }
        
        /* Light theme support */
        [data-theme="light"] .table-container {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(0, 0, 0, 0.06);
        }
        
        [data-theme="light"] .latex-table {
            background: rgba(255, 255, 255, 0.9);
        }
        
        [data-theme="light"] .latex-table thead th {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        }
        
        [data-theme="light"] .latex-table tbody tr {
            border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        }
        
        [data-theme="light"] .latex-table tbody tr:nth-child(even) {
            background: rgba(241, 245, 249, 0.7);
        }
        
        [data-theme="light"] .latex-table tbody tr:hover {
            background: rgba(99, 102, 241, 0.04);
        }
        
        [data-theme="light"] .latex-table td {
            color: #64748b;
        }
        
        [data-theme="light"] .latex-table td:first-child {
            color: #1e293b;
        }
        
        [data-theme="light"] .latex-table tbody tr:hover td {
            color: #374151;
        }
        
        [data-theme="light"] .table-caption {
            color: #374151;
        }
        
        /* Focus states for accessibility */
        .latex-table:focus-within {
            outline: 2px solid #6366f1;
            outline-offset: 2px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .table-container {
                margin: 16px 0;
                border-radius: 8px;
            }
            
            .latex-table {
                font-size: 13px;
            }
            
            .latex-table th, .latex-table td {
                padding: 12px 16px;
                min-width: 60px;
            }
            
            .latex-table thead th {
                font-size: 12px;
                padding: 14px 16px;
            }
        }
        
        @media (max-width: 480px) {
            .latex-table th, .latex-table td {
                padding: 10px 12px;
                font-size: 12px;
            }
            
            .table-caption {
                font-size: 12px;
                margin-top: 10px;
            }
        }
</style>
        """
        
        return css_style + ''.join(html_parts)
        
    except Exception as e:
        logger.error(f"Error converting LaTeX table: {str(e)}")
        return f"<p>Error processing table: {str(e)}</p>"


def restore_tables_in_response(text: str, tables: Dict[str, str]) -> str:
    """Restore HTML tables in the response text"""
    for table_id, html_table in tables.items():
        text = text.replace(f"[{table_id}]", html_table)
    return text


# Update the generate_prompt function in your FastAPI backend
def generate_prompt(
    subject_data: Dict, question: str, chat_history: List[Dict[str, str]]
) -> tuple[str, Dict[str, str]]:
    """Generate prompt with context-aware diagram instructions and LaTeX to plain text conversion"""
    syllabus_content = []
    diagram_map = {}  # Track diagrams with their context
    all_tables = {}  # Store all extracted tables

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
            
            # Extract tables and convert to placeholders
            text_with_placeholders, tables = extract_and_convert_latex_tables(topic["content"]["text"])
            all_tables.update(tables)
            
            syllabus_content.append(
                latex_to_plaintext(text_with_placeholders.strip())
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
                
                # Extract tables and convert to placeholders
                subtext_with_placeholders, subtables = extract_and_convert_latex_tables(subtopic["content"]["text"])
                all_tables.update(subtables)
                
                syllabus_content.append(
                    latex_to_plaintext(subtext_with_placeholders.strip())
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

    # Table placeholders for the prompt
    table_placeholders = list(all_tables.keys())
    table_instructions = ""
    if table_placeholders:
        table_instructions = f"""
**Available Table Placeholders**: {', '.join(table_placeholders)}
- These placeholders represent formatted tables from the syllabus
- Include relevant table placeholders in your response exactly as they appear
- DO NOT modify or recreate table content - use the placeholders as-is
- Tables will be automatically formatted in the final response
        """

    # Construct the strict prompt
    prompt = f"""
**Syllabus Content**:
{syllabus_text}
{history_text}

**Current Question**:
{latex_to_plaintext(question)}

**Available Diagrams with Context**:
{"\n".join(diagram_instructions) if diagram_instructions else "No diagrams available"}

{table_instructions}

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

**Table Inclusion Rules**:
1. If your response needs to reference tabular data from the syllabus, use the exact table placeholder
2. Place table placeholders where they are most relevant to your explanation
3. DO NOT create new tables or modify existing table content
4. Use placeholders exactly as provided: [TABLE_PLACEHOLDER_X]

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
    return prompt, all_tables


def latex_to_plaintext(text: str) -> str:
    """Convert LaTeX to plain text with special handling for lstlisting and math, but preserve table placeholders"""
    if not text:
        return text

    # Preserve table placeholders
    table_placeholders = re.findall(r'\[TABLE_PLACEHOLDER_\d+\]', text)
    
    # First process lstlisting blocks
    text = process_lstlisting_blocks(text)
    
    # Then handle math expressions (existing code)
    text = re.sub(r"\$(.*?)\$", r"\1", text)  # Inline math
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text)  # Display math
    text = re.sub(r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}", r"\1", text)
    text = re.sub(r"\\begin\{align\*\}(.*?)\\end\{align\*\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)  # Remove other LaTeX commands
    
    return text

def process_lstlisting_blocks(text: str) -> str:
    """Extract and format lstlisting blocks as markdown code blocks"""
    lstlisting_pattern = re.compile(
        r'\\begin\{lstlisting\}(\[.*?\])?(.*?)\\end\{lstlisting\}',
        re.DOTALL
    )
    
    def replace_lstlisting(match):
        options = match.group(1) or ''
        code_content = match.group(2).strip()
        
        # Extract language if specified
        language = 'text'
        lang_match = re.search(r'language=([a-zA-Z]+)', options)
        if lang_match:
            language = lang_match.group(1).lower()
        
        # Convert to markdown code block
        return f"```{language}\n{code_content}\n```"
    
    return lstlisting_pattern.sub(replace_lstlisting, text)


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

        # Generate context-aware prompt using cleaned history and get tables
        prompt, extracted_tables = generate_prompt(subject_data, request.question, cleaned_history)

        logger.info(
            f"Generated prompt: {prompt[:500]}..."
        )  # Log first 500 chars of prompt
        logger.info(f"Extracted {len(extracted_tables)} tables")

        # Get response from Gemini
        response = gemini_model.generate_content(prompt)

        # Get the text response
        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from AI")

        # Restore tables in the response
        final_response = restore_tables_in_response(response.text, extracted_tables)

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
                {"role": "assistant", "content": final_response},
            ]
        )

        # Limit chat history to last 10 messages
        chat_contexts[context_key] = chat_contexts[context_key][-10:]

        return {
            "answer": final_response,
            "sources": {
                "subject": request.subject,
                "department": request.department,
                "semester": request.semester,
            },
            "containsTable": len(extracted_tables) > 0,
            "tableCount": len(extracted_tables)
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