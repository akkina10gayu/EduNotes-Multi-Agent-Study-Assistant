"""
Streamlit UI for EduNotes
Version 2.0 - With Study Features (Flashcards, Quizzes, Progress)
"""
import streamlit as st
import requests
import json
import html as html_module
from datetime import datetime
import time
import random
import os
import re as re_module
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration (defined early for cached functions)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# =============================================================================
# DEBUG LOGGING FOR API CALLS
# =============================================================================
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
api_logger = logging.getLogger("edunotes.api_debug")
api_logger.setLevel(logging.DEBUG)


def _log_response(response, *args, **kwargs):
    """Hook that logs every HTTP response from the API session."""
    body = ""
    try:
        body = response.text[:500]
    except Exception:
        body = "<unreadable>"
    api_logger.info(
        "API %s %s -> %s | headers=%s | body=%s",
        response.request.method,
        response.request.url,
        response.status_code,
        dict(response.headers),
        body,
    )
    if response.status_code == 401:
        api_logger.error(
            ">>> 401 DETAIL: request_headers=%s | response_body=%s",
            dict(response.request.headers),
            body,
        )


# =============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE OPTIMIZATION (Phase 1)
# =============================================================================

@st.cache_resource
def get_api_session():
    """Create a reusable requests session for connection pooling.
    This reduces TCP handshake overhead for multiple API calls.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "EduNotes-UI/1.0 (Streamlit; internal)",
        "X-Requested-With": "EduNotes-Streamlit",
    })
    session.hooks["response"].append(_log_response)
    return session


def check_api_health():
    """Check if API is healthy with smart caching.
    - Successful results cached for 30 seconds
    - Failed results cached for only 5 seconds (quick retry)
    This prevents long-cached failures from showing 'offline' when API is actually running.
    """
    cache_key = '_api_health_cache'
    cache_time_key = '_api_health_cache_time'

    current_time = time.time()

    # Check if we have a valid cached result
    if cache_key in st.session_state and cache_time_key in st.session_state:
        cached_value = st.session_state[cache_key]
        cached_time = st.session_state[cache_time_key]

        # Different TTL: 30s for success, 30s for failure (avoid retry storms)
        ttl = 30 if cached_value else 30

        if current_time - cached_time < ttl:
            return cached_value

    # Cache expired or not set - make the API call
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/health", timeout=60)  # Increased from 2s
        result = response.status_code == 200
    except:
        result = False

    # Cache the result
    st.session_state[cache_key] = result
    st.session_state[cache_time_key] = current_time

    return result


@st.cache_data(ttl=300)
def fetch_topics():
    """Fetch topics from API - cached for 5 minutes.
    Topics rarely change, so 5 minute cache is appropriate.
    Returns list of topic strings or empty list on failure.
    """
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/topics", timeout=60)  # Phase 4: reduced from 3s
        if response.status_code == 200:
            return response.json().get('topics', [])
    except:
        pass
    return []


def fetch_study_stats():
    """Fetch study progress and flashcard stats with smart caching.
    - Successful results cached for 60 seconds
    - Failed results cached for only 5 seconds (quick retry)
    Combines two API calls into one cached result.
    Returns dict with stats or None on failure.
    """
    cache_key = '_study_stats_cache'
    cache_time_key = '_study_stats_cache_time'

    current_time = time.time()

    # Check if we have a valid cached result
    if cache_key in st.session_state and cache_time_key in st.session_state:
        cached_value = st.session_state[cache_key]
        cached_time = st.session_state[cache_time_key]

        # Different TTL: 60s for success, 30s for failure (avoid retry storms)
        ttl = 60 if cached_value is not None else 30

        if current_time - cached_time < ttl:
            return cached_value

    # Cache expired or not set - make the API calls
    try:
        session = get_api_session()
        progress_resp = session.get(f"{API_BASE_URL}/study/progress", timeout=60)  # Increased from 2s
        flashcard_resp = session.get(f"{API_BASE_URL}/study/flashcards/sets", timeout=60)  # Increased from 2s

        result = {
            'total_flashcards': 0,
            'total_quizzes': 0,
            'current_streak': 0,
            'longest_streak': 0,
            'flashcard_sets': 0
        }

        if progress_resp.status_code == 200:
            progress_data = progress_resp.json()
            overall_stats = progress_data.get('overall_stats', {})
            streak_info = progress_data.get('streak', {})
            result['total_flashcards'] = overall_stats.get('total_flashcards_reviewed', 0)
            result['total_quizzes'] = overall_stats.get('total_quizzes_completed', 0)
            result['current_streak'] = streak_info.get('current_streak', 0)
            result['longest_streak'] = streak_info.get('longest_streak', 0)

        if flashcard_resp.status_code == 200:
            flashcard_data = flashcard_resp.json()
            result['flashcard_sets'] = len(flashcard_data.get('sets', []))
    except:
        result = None

    # Cache the result
    st.session_state[cache_key] = result
    st.session_state[cache_time_key] = current_time

    return result


# =============================================================================
# PHASE 2 - CACHED FUNCTIONS FOR STUDY MODE (Lazy Loading)
# =============================================================================

@st.cache_data(ttl=30)
def fetch_flashcard_sets():
    """Fetch flashcard sets - cached for 30 seconds.
    Used by both 'Load Existing Set' and 'Export to Anki' sections.
    Eliminates duplicate API calls within Flashcards sub-tab.
    Returns list of sets or empty list on failure.
    """
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/study/flashcards/sets", timeout=60)  # Phase 4: reduced from 5s
        if response.status_code == 200:
            return response.json().get('sets', [])
    except:
        pass
    return []


@st.cache_data(ttl=30)
def fetch_quizzes_list():
    """Fetch quizzes list - cached for 30 seconds.
    Used by 'Load Existing Quiz' section.
    Returns list of quizzes or empty list on failure.
    """
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/study/quizzes", timeout=60)  # Phase 4: reduced from 5s
        if response.status_code == 200:
            return response.json().get('quizzes', [])
    except:
        pass
    return []


@st.cache_data(ttl=60)
def fetch_detailed_progress():
    """Fetch detailed study progress - cached for 60 seconds.
    Returns full progress data for Progress Dashboard.
    Includes: overall_stats, streak, topic_rankings, weekly_summary, recent_activities.
    Returns dict or None on failure.
    """
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/study/progress", timeout=60)  # Phase 4: reduced from 5s
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_kb_documents(keyword=None):
    """Fetch document list from KB - cached for 60 seconds.
    Returns list of document metadata (no content) or empty list on failure.
    """
    try:
        session = get_api_session()
        params = {}
        if keyword:
            params['keyword'] = keyword
        response = session.get(f"{API_BASE_URL}/documents", params=params, timeout=60)
        if response.status_code == 200:
            return response.json().get('documents', [])
    except:
        pass
    return []


@st.cache_data(ttl=60)
def fetch_kb_documents_semantic(query):
    """Semantic search for documents via vector DB chunk matching.
    Returns list of document metadata (no content) or empty list on failure.
    """
    try:
        session = get_api_session()
        response = session.get(
            f"{API_BASE_URL}/documents/search",
            params={"query": query, "k": 20},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('documents', [])
    except:
        pass
    return []


def fetch_document_content(doc_id):
    """Fetch full document text by ID with smart caching.
    - Successful results cached for 5 minutes
    - Failed results cached for 5 seconds
    Returns dict with document data or None on failure.
    """
    cache_key = f'_doc_content_{doc_id}'
    cache_time_key = f'_doc_content_time_{doc_id}'

    current_time = time.time()

    if cache_key in st.session_state and cache_time_key in st.session_state:
        cached_value = st.session_state[cache_key]
        cached_time = st.session_state[cache_time_key]
        ttl = 300 if cached_value is not None else 5
        if current_time - cached_time < ttl:
            return cached_value

    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/documents/{doc_id}", timeout=60)
        if response.status_code == 200:
            result = response.json().get('document')
        else:
            result = None
    except:
        result = None

    st.session_state[cache_key] = result
    st.session_state[cache_time_key] = current_time
    return result


def fetch_system_stats():
    """Fetch system stats with smart caching.
    - Successful results cached for 2 minutes
    - Failed results cached for only 5 seconds (quick retry)
    System stats rarely change, so longer cache is appropriate for successes.
    Returns dict or None on failure.
    """
    cache_key = '_system_stats_cache'
    cache_time_key = '_system_stats_cache_time'

    current_time = time.time()

    # Check if we have a valid cached result
    if cache_key in st.session_state and cache_time_key in st.session_state:
        cached_value = st.session_state[cache_key]
        cached_time = st.session_state[cache_time_key]

        # Different TTL: 120s for success, 30s for failure (avoid retry storms)
        ttl = 120 if cached_value is not None else 30

        if current_time - cached_time < ttl:
            return cached_value

    # Cache expired or not set - make the API call
    try:
        session = get_api_session()
        response = session.get(f"{API_BASE_URL}/stats", timeout=60)  # Increased from 2s
        if response.status_code == 200:
            result = response.json()
        else:
            result = None
    except:
        result = None

    # Cache the result
    st.session_state[cache_key] = result
    st.session_state[cache_time_key] = current_time

    return result


# =============================================================================
# RATE LIMIT HANDLING
# =============================================================================

def _is_rate_limit_error(error_msg: str) -> bool:
    """Check if an error message indicates a Groq API rate limit."""
    if not error_msg:
        return False
    msg = str(error_msg).lower()
    return 'rate limit' in msg or 'rate_limit' in msg or ('429' in str(error_msg) and 'token' in msg)


def _show_rate_limit_dialog(error_msg: str, key_suffix: str = ""):
    """Show rate limit warning with options to wait or switch model."""
    st.warning("**API Rate Limit Reached** — Daily token quota exceeded for the current Groq model.")

    # Extract wait time from error
    wait_match = re_module.search(r'try again in ([\dhmins. ]+)', str(error_msg), re_module.IGNORECASE)
    if wait_match:
        st.info(f"Estimated reset in: **{wait_match.group(1).strip()}**")

    option = st.radio(
        "Choose an option:",
        ["Wait for rate limit to reset", "Switch to a different model"],
        key=f"rate_limit_option{key_suffix}",
        horizontal=True
    )

    if option == "Switch to a different model":
        FALLBACK_MODELS = {
            "llama-3.3-70b-versatile": "LLaMA 3.3 70B (Best quality, 12K TPM, 100K TPD)",
            "llama-3.1-8b-instant": "LLaMA 3.1 8B (Fast, 500K TPD, high request limit)",
            "qwen/qwen3-32b": "Qwen3 32B (Good quality, 500K TPD)",
        }
        selected = st.selectbox(
            "Select alternate model:",
            list(FALLBACK_MODELS.keys()),
            format_func=lambda x: FALLBACK_MODELS[x],
            key=f"rate_limit_model{key_suffix}"
        )
        if st.button("Switch Model", key=f"rate_limit_switch{key_suffix}"):
            try:
                env_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'
                )
                with open(env_path, 'r') as f:
                    env_content = f.read()
                env_content = re_module.sub(r'GROQ_MODEL=.*', f'GROQ_MODEL={selected}', env_content)
                with open(env_path, 'w') as f:
                    f.write(env_content)
                st.success(f"Model switched to **{selected}**")
                st.info("Please **restart the API server** for the change to take effect:\n```\nuvicorn src.api.app:app --reload\n```")
            except Exception as ex:
                st.error(f"Failed to update configuration: {ex}")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="EduNotes - Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for note history and UI controls
if 'note_history' not in st.session_state:
    st.session_state.note_history = []
if 'selected_history' not in st.session_state:
    st.session_state.selected_history = None
if 'show_copy_box' not in st.session_state:
    st.session_state.show_copy_box = False
if 'show_copy_box_history' not in st.session_state:
    st.session_state.show_copy_box_history = False
if 'font_size' not in st.session_state:
    st.session_state.font_size = 'medium'  # Default font size
if 'first_time_user' not in st.session_state:
    st.session_state.first_time_user = True
if 'dismissed_welcome' not in st.session_state:
    st.session_state.dismissed_welcome = False
if 'show_system_stats' not in st.session_state:
    st.session_state.show_system_stats = False
if 'api_status_refresh' not in st.session_state:
    st.session_state.api_status_refresh = 0
if 'last_generated_notes' not in st.session_state:
    st.session_state.last_generated_notes = None  # Stores the last generated notes
if 'notes_generated_time' not in st.session_state:
    st.session_state.notes_generated_time = None  # Timestamp of generation
if 'last_generation_metadata' not in st.session_state:
    st.session_state.last_generation_metadata = None  # Metadata (query, type, sources)
if 'last_llm_provider' not in st.session_state:
    st.session_state.last_llm_provider = None  # Track last used LLM provider
if 'last_llm_model' not in st.session_state:
    st.session_state.last_llm_model = None  # Track last used LLM model

# Phase 5: PDF caching - store last processed PDF to avoid re-extraction
if 'last_pdf_name' not in st.session_state:
    st.session_state.last_pdf_name = None
if 'last_pdf_size' not in st.session_state:
    st.session_state.last_pdf_size = None
if 'last_pdf_text' not in st.session_state:
    st.session_state.last_pdf_text = None  # Extracted text from PDF

# Vision data for Research Mode figure expander
if 'last_vision_data' not in st.session_state:
    st.session_state.last_vision_data = None

# Summarization mode - initialize to ensure consistent state across reruns
if 'summarization_mode_selector' not in st.session_state:
    st.session_state.summarization_mode_selector = 'paragraph_summary'

# Save to KB feature - show/hide form state and saved status
if 'show_save_kb_recent' not in st.session_state:
    st.session_state.show_save_kb_recent = False
if 'show_save_kb_history' not in st.session_state:
    st.session_state.show_save_kb_history = False
if 'saved_to_kb_recent' not in st.session_state:
    st.session_state.saved_to_kb_recent = False
if 'saved_to_kb_history' not in st.session_state:
    st.session_state.saved_to_kb_history = False

# Edit mode for notes (inline editing with undo support)
if 'edit_mode_recent' not in st.session_state:
    st.session_state.edit_mode_recent = False
if 'edit_undo_recent' not in st.session_state:
    st.session_state.edit_undo_recent = None
if 'edit_mode_history' not in st.session_state:
    st.session_state.edit_mode_history = False
if 'edit_undo_history' not in st.session_state:
    st.session_state.edit_undo_history = None

# Undo/redo stacks for inline text editing
if 'edit_undo_stack_recent' not in st.session_state:
    st.session_state.edit_undo_stack_recent = []
if 'edit_redo_stack_recent' not in st.session_state:
    st.session_state.edit_redo_stack_recent = []
if 'edit_current_text_recent' not in st.session_state:
    st.session_state.edit_current_text_recent = None
if 'edit_undo_stack_history' not in st.session_state:
    st.session_state.edit_undo_stack_history = []
if 'edit_redo_stack_history' not in st.session_state:
    st.session_state.edit_redo_stack_history = []
if 'edit_current_text_history' not in st.session_state:
    st.session_state.edit_current_text_history = None

def _on_edit_change_recent():
    """Callback for recent notes text_area on_change — captures edit snapshots."""
    new_text = st.session_state.edit_textarea_recent
    current = st.session_state.edit_current_text_recent
    if current is not None and new_text != current:
        st.session_state.edit_undo_stack_recent.append(current)
        st.session_state.edit_redo_stack_recent.clear()
        st.session_state.edit_current_text_recent = new_text

def _on_edit_change_history():
    """Callback for history notes text_area on_change — captures edit snapshots."""
    new_text = st.session_state.edit_textarea_history
    current = st.session_state.edit_current_text_history
    if current is not None and new_text != current:
        st.session_state.edit_undo_stack_history.append(current)
        st.session_state.edit_redo_stack_history.clear()
        st.session_state.edit_current_text_history = new_text

def _undo_edit_recent():
    """on_click callback for recent notes undo button. Runs before widgets render."""
    undo_stack = st.session_state.edit_undo_stack_recent
    if undo_stack:
        st.session_state.edit_redo_stack_recent.append(st.session_state.edit_current_text_recent)
        prev = undo_stack.pop()
        st.session_state.edit_current_text_recent = prev
        st.session_state.edit_textarea_recent = prev

def _redo_edit_recent():
    """on_click callback for recent notes redo button. Runs before widgets render."""
    redo_stack = st.session_state.edit_redo_stack_recent
    if redo_stack:
        st.session_state.edit_undo_stack_recent.append(st.session_state.edit_current_text_recent)
        next_state = redo_stack.pop()
        st.session_state.edit_current_text_recent = next_state
        st.session_state.edit_textarea_recent = next_state

def _undo_edit_history():
    """on_click callback for history notes undo button. Runs before widgets render."""
    undo_stack = st.session_state.edit_undo_stack_history
    if undo_stack:
        st.session_state.edit_redo_stack_history.append(st.session_state.edit_current_text_history)
        prev = undo_stack.pop()
        st.session_state.edit_current_text_history = prev
        st.session_state.edit_textarea_history = prev

def _redo_edit_history():
    """on_click callback for history notes redo button. Runs before widgets render."""
    redo_stack = st.session_state.edit_redo_stack_history
    if redo_stack:
        st.session_state.edit_undo_stack_history.append(st.session_state.edit_current_text_history)
        next_state = redo_stack.pop()
        st.session_state.edit_current_text_history = next_state
        st.session_state.edit_textarea_history = next_state

# Font size mapping
font_sizes = {
    'small': {'base': '15px', 'header': '1.4rem', 'notes': '0.95rem'},
    'medium': {'base': '17px', 'header': '1.55rem', 'notes': '1.05rem'},
    'large': {'base': '20px', 'header': '1.7rem', 'notes': '1.15rem'}
}

current_size = font_sizes[st.session_state.font_size]

# Theme is handled entirely by Streamlit's native Settings panel +
# .streamlit/config.toml for persistence. No CSS injection needed.

# Apply unified CSS with dynamic font sizing
st.markdown(f"""
<style>
    /* Base styling */
    html, body, .stApp {{
        font-size: {current_size['base']};
    }}

    .main-header {{
        font-size: {current_size['header']} !important;
        color: #6CA0DC;
        text-align: center;
        font-weight: 700;
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0.75rem;
    }}

    .success-message {{
        padding: 1rem;
        background-color: #1B3A1B;
        border-radius: 0.5rem;
        color: #4CAF50;
    }}

    .error-message {{
        padding: 1rem;
        background-color: #3A1B1B;
        border-radius: 0.5rem;
        color: #F44336;
    }}

    .notes-container {{
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #6CA0DC;
        font-size: {current_size['notes']};
        color: #E8E8E8;
    }}
    /* Ensure all text inside the dark notes container stays light */
    .notes-container h1, .notes-container h2, .notes-container h3,
    .notes-container h4, .notes-container h5, .notes-container h6,
    .notes-container p, .notes-container li, .notes-container strong,
    .notes-container code, .notes-container a {{
        color: #E8E8E8;
    }}
    .notes-container a {{
        color: #6CA0DC;
    }}
    /* Notes headings — just slightly larger than body text */
    .notes-container h1 {{
        font-size: 1.08rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.15rem;
    }}
    .notes-container h2 {{
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.12rem;
    }}
    .notes-container h3 {{
        font-size: 1.02rem;
        font-weight: 600;
        margin-top: 0.3rem;
        margin-bottom: 0.08rem;
    }}

    /* Improve readability */
    .stMarkdown p, .stText {{
        line-height: 1.6;
    }}

    /* Button styling */
    .stButton>button {{
        font-size: {current_size['base']};
    }}

    /* Hide the "Press Ctrl+Enter to apply" instruction on text areas.
       All text areas are paired with action buttons (Save, Generate, etc.)
       that commit the value via focus loss, making this label unnecessary
       and confusing for users. */
    .stTextArea [data-testid="InputInstructions"] {{
        display: none;
    }}

    /* Tighten spacing between radio button groups (Search Mode, Output Format, Summary Length) */
    .stRadio {{
        margin-bottom: -0.5rem;
    }}
    .stRadio > div {{
        gap: 0.3rem;
    }}

    /* Selectbox dropdown — limit height so it opens below, not above */
    [data-baseweb="menu"] {{
        max-height: 300px !important;
        overflow-y: auto !important;
    }}

    /* Compact Study Stats metrics */
    [data-testid="stMetricValue"] {{
        font-size: 1.2rem;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.8rem;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.7rem;
    }}

    /* Tighter dividers / horizontal rules */
    hr {{
        margin: 0.5rem 0 !important;
    }}

    /* Compact tab panel content */
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 0.5rem;
    }}

    /* --- Reduce top padding above EduNotes header --- */
    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 0rem !important;
    }}

    /* Minimize Streamlit default header bar space */
    header[data-testid="stHeader"] {{
        height: 2.2rem !important;
        min-height: 0 !important;
    }}

    /* --- Font Hierarchy (sizes & weights only — colors follow theme) --- */

    /* Tab labels — prominent and readable */
    .stTabs [data-baseweb="tab"] button {{
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.1rem !important;
    }}

    /* h3 headings — section headers like "Recently Generated Notes" */
    .stMarkdown h3 {{
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }}

    /* h4 headings — sub-sections like "Save Notes to KB" */
    .stMarkdown h4 {{
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }}

    /* Bold section text — inline headers like "Study Stats" */
    .stMarkdown strong {{
        font-size: 1.08rem;
    }}

    /* Widget labels — consistent size/weight across all input types */
    .stTextArea label p,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    .stRadio > label > div > p,
    .stTextInput label p,
    .stSelectbox label p {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }}

    /* File uploader dropzone: align label */
    [data-testid="stFileUploader"] section {{
        margin-top: 0;
    }}

    /*
     * Theme-adaptive text colors using var(--text-color).
     * Streamlit sets --text-color to dark/black in light mode
     * and white/light in dark mode, so these rules adapt
     * automatically when the user switches themes.
     */

    /* --- All text throughout the UI follows theme --- */
    .stTextArea textarea,
    .stTextInput input {{
        color: var(--text-color) !important;
    }}
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {{
        color: var(--text-color) !important;
        opacity: 0.45;
    }}

    /* File uploader dropzone text */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {{
        color: var(--text-color) !important;
    }}
    [data-testid="stFileUploader"] section small {{
        opacity: 0.6;
    }}

    /* --- Sidebar text follows theme --- */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stHeading"],
    [data-testid="stSidebar"] [data-testid="stHeading"] *,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label div,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] [data-testid="stExpanderToggleDetails"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary div {{
        color: var(--text-color) !important;
    }}

    /* Sidebar expander arrow/icon */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }}

    /* Sidebar buttons — theme-adaptive text and subtle borders */
    [data-testid="stSidebar"] .stButton > button {{
        border: 1px solid rgba(0, 0, 0, 0.15) !important;
        color: var(--text-color) !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: rgba(0, 0, 0, 0.05) !important;
    }}

    /* Sidebar expanders — subtle border for definition */
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 0.5rem;
    }}

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {{
        border-color: rgba(0, 0, 0, 0.1) !important;
    }}

    /* --- Captions: inherit theme text color, slightly muted --- */
    [data-testid="stCaptionContainer"] p {{
        color: var(--text-color) !important;
        opacity: 0.75;
    }}

    /* --- All buttons: ensure text follows theme --- */
    .stButton > button {{
        color: var(--text-color) !important;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h2 class="main-header">📚 EduNotes Study Assistant</h2>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # API Status Check - Using cached function (30 sec TTL)
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.caption("Start API: uvicorn src.api.app:app --reload")

    st.divider()

    # Font Size Control
    st.markdown("#### 📏 Text Size")
    font_choice = st.radio(
        "Adjust text size for better readability:",
        options=["Small", "Medium", "Large"],
        index=["small", "medium", "large"].index(st.session_state.font_size),
        key="font_size_radio",
        horizontal=True,
        label_visibility="collapsed"
    )

    # Update state if changed
    new_font_size = font_choice.lower()
    if new_font_size != st.session_state.font_size:
        st.session_state.font_size = new_font_size
        st.rerun()

    st.divider()

    # System Stats - Using expander with cached function (Phase 4 optimization)
    with st.expander("📊 System Stats", expanded=False):
        stats = fetch_system_stats()  # Uses cached function (2 min TTL)

        if stats:
            # LLM Info
            llm_stats = stats.get('llm', {})
            provider = llm_stats.get('provider', 'unknown')
            model = llm_stats.get('model', 'unknown')
            available = llm_stats.get('available', False)

            st.markdown("**🧠 LLM Model**")
            st.markdown(f"- Mode: ☁️ Cloud API")
            st.markdown(f"- Provider: {provider.title() if provider else 'None'}")
            st.markdown(f"- Model: {model}")
            if not available:
                st.markdown("- Status: ⚠️ No provider available")

            # KB Info
            kb_stats = stats.get('knowledge_base', {})
            st.markdown("**📚 Knowledge Base**")
            st.markdown(f"- Documents: {kb_stats.get('total_documents', 0):,}")
            st.markdown(f"- Collection: {kb_stats.get('collection_name', 'N/A')}")

            # Agents Info
            st.markdown("**🤖 AI Agents**")
            agents = stats.get('agents', {})
            for agent_name, status in agents.items():
                status_icon = "🟢" if status == "active" else "🔴"
                agent_display = agent_name.replace('_', ' ').title()
                st.markdown(f"- {agent_display}: {status_icon} {status.title()}")
        else:
            st.warning("⚠️ Could not load system stats")
            st.caption("API may be offline. Run: uvicorn src.api.app:app --reload")

    st.divider()

    # Note History - Collapsible with native Streamlit components
    with st.expander("📜 Note History", expanded=False):
        if st.session_state.note_history:
            # Filter out notes that match what's currently displayed in "Recently Generated Notes"
            # to avoid showing the same notes in both places
            # Use content-based filtering instead of position-based to handle Streamlit's execution order
            current_notes = st.session_state.get('last_generated_notes')
            if current_notes is not None:
                # Filter out entries where notes content matches the currently displayed notes
                # This prevents duplication between "Recently Generated Notes" and history
                filtered_history = [
                    entry for entry in st.session_state.note_history
                    if entry.get('notes') != current_notes
                ]
            else:
                # No notes currently displayed, show all history
                filtered_history = st.session_state.note_history

            if filtered_history:
                st.caption(f"Last {len(filtered_history)} notes")

                # Display history items using native Streamlit (most recent first)
                for idx, note_item in enumerate(reversed(filtered_history)):
                    # Truncate query for display
                    query_short = note_item['query'][:35] + "..." if len(note_item['query']) > 35 else note_item['query']
                    type_icon = {"topic": "🎯", "url": "🔗", "text": "📄", "pdf": "📑"}.get(note_item['type'], "📝")

                    # Create a compact display for each history item
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{type_icon} {query_short}**")
                        st.caption(f"{note_item['timestamp']} • {note_item['type'].upper()}")
                    with col2:
                        if st.button("View", key=f"history_view_{idx}", use_container_width=True):
                            st.session_state.selected_history = note_item
                            st.session_state.saved_to_kb_history = False  # Reset save state for new selection
                            st.session_state.edit_mode_history = False
                            st.session_state.edit_undo_history = None
                            st.session_state.edit_undo_stack_history = []
                            st.session_state.edit_redo_stack_history = []
                            st.session_state.edit_current_text_history = None
                            st.rerun()

                    # Add separator between items (except last)
                    if idx < len(filtered_history) - 1:
                        st.markdown("---")
            else:
                st.info("📝 Current notes are shown above. Close them to see history.")
        else:
            st.info("📝 No notes yet. Generate a note to see it here!")

    st.divider()

    # Help Section at Bottom
    with st.expander("❓ Help & Setup Guide"):
        st.markdown("### 🚀 Getting Started")

        st.markdown("#### 1️⃣ Start the API Server")
        st.code("uvicorn src.api.app:app --reload", language="bash")
        st.caption("Run this command in your terminal from the project root")

        st.markdown("#### 2️⃣ Start the UI")
        st.code("streamlit run ui/streamlit_app.py", language="bash")
        st.caption("Run this in a separate terminal")

        st.markdown("---")

        st.markdown("### ⚡ Setup API Key (Recommended)")
        st.markdown("""
        **Get FREE Groq API Key for 10x faster performance:**

        1. Visit [console.groq.com](https://console.groq.com)
        2. Sign up (no credit card required)
        3. Copy your API key
        4. Edit `.env` file in project root:
        """)
        st.code("GROQ_API_KEY=your_key_here", language="bash")
        st.markdown("5. Restart both API and UI")

        st.markdown("---")

        st.markdown("### 📚 How to Use")
        st.markdown("""
        **Generate Notes:**
        - Enter a topic (e.g., "Machine Learning")
        - Paste a URL to an article
        - Upload a PDF file
        - Paste text directly

        **Study Features:**
        - Create flashcards from notes
        - Take quizzes to test knowledge
        - Track your progress and streaks
        - Export flashcards to Anki

        **Tips:**
        - Click topic chips for quick queries
        - Use note history to revisit past notes
        - Copy notes directly without downloading
        """)

        st.markdown("---")

        st.markdown("### ⚙️ Troubleshooting")
        st.markdown("""
        **API Offline?**
        - Make sure you ran `uvicorn src.api.app:app --reload`
        - Check if port 8000 is available
        - Look at terminal for error messages

        **Slow Performance?**
        - Add a Groq API key (see above)
        - Local models are slower but work offline

        **PDF Not Processing?**
        - Make sure file is less than 10MB
        - PDF must have readable text (not images only)
        """)

# First-Time User Welcome Banner
if st.session_state.first_time_user and not st.session_state.dismissed_welcome:
    st.info("👋 **Welcome to EduNotes!** New here? Check out the **Help & Setup Guide** in the sidebar (bottom left) to get started.")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Got it! ✓", key="dismiss_welcome", use_container_width=True):
            st.session_state.dismissed_welcome = True
            st.session_state.first_time_user = False
            st.rerun()

# Quick Stats Dashboard - Using cached function (60 sec TTL)
st.markdown("**📊 Study Stats**")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

notes_generated = len(st.session_state.get('note_history', []))
study_stats = fetch_study_stats() if check_api_health() else None

if study_stats:
    with stat_col1:
        st.metric("📝 Notes", notes_generated, help="Session")
    with stat_col2:
        st.metric("🃏 Flashcards", study_stats['flashcard_sets'], help="Total sets")
    with stat_col3:
        st.metric("📋 Quizzes", study_stats['total_quizzes'], help="Total taken")
    with stat_col4:
        # Show current streak with best streak as delta indicator
        current = study_stats['current_streak']
        best = study_stats['longest_streak']
        streak_display = f"{current}" if current == best else f"{current}"
        st.metric("🔥 Streak", f"{streak_display} days", delta=f"Best: {best}", delta_color="off")
else:
    with stat_col1:
        st.metric("📝 Notes", notes_generated, help="Session")
    with stat_col2:
        st.metric("🃏 Flashcards", "—", help="Total sets")
    with stat_col3:
        st.metric("📋 Quizzes", "—", help="Total taken")
    with stat_col4:
        st.metric("🔥 Streak", "— days", delta="Best: —", delta_color="off")

st.divider()

# Main content area
st.markdown("""<style>
    button[data-baseweb="tab"] > div > p {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
</style>""", unsafe_allow_html=True)
tab1, tab_chat, tab2, tab3, tab4 = st.tabs(["📝 Generate Notes", "💬 Chat with AI", "🔍 Search Knowledge Base", "📤 Update Knowledge Base", "📖 Study Mode"])

# Tab 1: Generate Notes
with tab1:
    # Topic Suggestions - Using cached function (5 min TTL)
    st.markdown("**💡 Quick Topics**")

    # Initialize show_more_topics state
    if 'show_more_topics' not in st.session_state:
        st.session_state.show_more_topics = False

    # Default popular topics (always available as fallback)
    default_topics = [
        "Machine Learning", "Deep Learning", "Neural Networks",
        "Natural Language Processing", "Python", "Data Science",
        "Statistics", "Artificial Intelligence"
    ]

    # Fetch topics from KB (cached)
    kb_topics = fetch_topics()

    # Merge: KB topics first, then fill with defaults (no duplicates)
    all_topics = []
    seen = set()
    for topic in kb_topics + default_topics:
        if topic not in seen:
            all_topics.append(topic)
            seen.add(topic)

    # Determine how many to show (5 default = 1 row, 10 expanded = 2 rows)
    initial_count = 5
    expanded_count = 10
    show_count = expanded_count if st.session_state.show_more_topics else initial_count
    topics_to_show = all_topics[:show_count]

    # Display as clickable chips (5 per row, truncate long names)
    chip_cols = st.columns(5)
    for idx, topic in enumerate(topics_to_show):
        col_idx = idx % 5
        with chip_cols[col_idx]:
            # Truncate long topic names; show full name on hover via help tooltip
            max_chars = 14
            if len(topic) > max_chars:
                display = f"📚 {topic[:max_chars - 1]}…"
                tooltip = topic
            else:
                display = f"📚 {topic}"
                tooltip = None
            if st.button(display, key=f"topic_{idx}", use_container_width=True, help=tooltip):
                st.session_state.query_text_area = topic
                st.rerun()

    # Show more/less button if there are more topics
    if len(all_topics) > initial_count:
        if st.session_state.show_more_topics:
            if st.button("Show less", key="show_less_topics"):
                st.session_state.show_more_topics = False
                st.rerun()
        else:
            if st.button("Show more", key="show_more_topics_btn"):
                st.session_state.show_more_topics = True
                st.rerun()

    st.markdown("---")

    # Input section — query and PDF side by side
    input_col1, input_col2 = st.columns([3, 2])

    with input_col1:
        query_input = st.text_area(
            "Enter topic, URL, or paste text:",
            height=100,
            key="query_text_area",
            placeholder="Examples:\n- Machine Learning\n- https://example.com/article\n- Paste your text here..."
        )
        if query_input:
            if query_input.startswith(('http://', 'https://', 'www.')):
                st.caption("🔗 URL detected")
            elif len(query_input) > 500:
                st.caption("📄 Text detected")
            else:
                st.caption("🎯 Topic detected")

    with input_col2:
        uploaded_file = st.file_uploader(
            "📄 Or upload a PDF",
            type=['pdf'],
            help="Research papers, textbooks, any PDF (max 10MB)"
        )
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.caption(f"📎 {uploaded_file.name} ({file_size_mb:.2f}MB)")

    # Research Mode Toggle — always visible
    research_mode = st.toggle(
        "Research Mode",
        value=False,
        help="**Research Mode** enhances output with: academic paper discovery (Semantic Scholar + arXiv), "
             "better PDF extraction (tables, equations, structure preserved), "
             "and AI vision analysis of figures in PDFs.",
        key="research_mode_toggle"
    )
    if research_mode:
        st.caption("Research Mode ON — Enhanced extraction + related papers + figure analysis")

    # Search Mode Selection — hidden by default, shown only when topic input is detected
    search_mode = "auto"  # Default for non-topic queries

    _is_topic_input = False  # Hidden until topic input is actively detected
    if not uploaded_file and query_input and query_input.strip():
        _stripped = query_input.strip()
        if not _stripped.startswith(('http://', 'https://', 'www.')) and len(_stripped) <= 500:
            _is_topic_input = True

    if _is_topic_input:
        st.markdown("---")
        search_mode = st.radio(
            "🔍 Search Mode",
            options=["auto", "kb_only", "web_search", "both"],
            format_func=lambda x: {
                "auto": "🔄 Auto (Recommended)",
                "kb_only": "📚 Knowledge Base Only",
                "web_search": "🌐 Web Search Only",
                "both": "🔗 KB + Web Search"
            }.get(x, x),
            help="""**Auto**: Searches KB first, falls back to web search if no results.
**Knowledge Base Only**: Only searches local KB. No internet access.
**Web Search Only**: Searches the internet directly. Skips the KB.
**KB + Web Search**: Searches both KB and web, combines results.""",
            horizontal=True,
            key="search_mode_selector"
        )

    st.markdown("---")

    # Summarization Mode Selection
    summarization_mode = st.radio(
        "⚙️ Output Format",
        options=["paragraph_summary", "important_points", "key_highlights"],
        format_func=lambda x: {
            "paragraph_summary": "📖 Paragraph Summary",
            "important_points": "📋 Important Points",
            "key_highlights": "⚡ Key Highlights"
        }.get(x, x),
        help="""**Paragraph Summary**: Flowing paragraphs with full context and relationships.
**Important Points**: Numbered key points, each independent and unique.
**Key Highlights**: Essential terms with brief definitions for quick scanning.""",
        horizontal=True,
        key="summarization_mode_selector"
    )

    # Output Length Selection - ONLY shown for Paragraph Summary
    output_length = "auto"  # Default value
    # Use session state directly after widget render to ensure correct value during reruns
    if st.session_state.summarization_mode_selector == "paragraph_summary":
        output_length = st.radio(
            "📏 Summary Length",
            options=["auto", "brief", "medium", "detailed"],
            format_func=lambda x: {
                "auto": "🔄 Auto (Recommended)",
                "brief": "📝 Brief (5-8 lines)",
                "medium": "📄 Medium (2-3 paragraphs)",
                "detailed": "📚 Detailed (4-6 paragraphs)"
            }.get(x, x),
            help="""**Auto**: Adjusts length based on input size automatically.
**Brief**: Quick 5-8 line overview of key takeaways.
**Medium**: Balanced 2-3 paragraph summary with moderate detail.
**Detailed**: Comprehensive 4-6 paragraph explanation with full coverage.""",
            horizontal=True,
            key="output_length_selector"
        )

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate Notes", type="primary"):
        # Check for conflict: both PDF and text provided
        if uploaded_file and query_input and query_input.strip():
            st.warning("⚠️ **Please choose one input method**")
            st.info("You have both a PDF attached and text entered. Please either:\n"
                    "- **Remove the PDF** (click ✕ on the file) to use text input, OR\n"
                    "- **Clear the text box** to process the PDF")
            st.stop()

        # Check if either query or PDF is provided
        if uploaded_file:
            # Process PDF file
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Validate file size
                pdf_content = uploaded_file.getvalue()
                file_size_mb = len(pdf_content) / (1024 * 1024)
                current_pdf_name = uploaded_file.name
                current_pdf_size = len(pdf_content)

                if file_size_mb > 10:
                    st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is 10MB.")
                else:
                    # Phase 5: Check if same PDF as last time (use cached text)
                    cached_text = st.session_state.last_pdf_text
                    use_cached_text = (
                        st.session_state.last_pdf_name == current_pdf_name and
                        st.session_state.last_pdf_size == current_pdf_size and
                        cached_text is not None
                    )

                    mode_text_map = {
                        "paragraph_summary": "paragraph summary",
                        "important_points": "important points",
                        "key_highlights": "key highlights"
                    }
                    mode_text = mode_text_map.get(summarization_mode, "summary")

                    if use_cached_text:
                        # Use cached extracted text - skip PDF extraction (internal optimization)
                        progress_placeholder.progress(0.3)
                        status_placeholder.info(f"🤖 Processing PDF content ({mode_text})...")

                        # Send cached text to /process-pdf as form data (skips extraction)
                        response = get_api_session().post(
                            f"{API_BASE_URL}/process-pdf",
                            data={
                                "summarization_mode": summarization_mode,
                                "output_length": output_length,
                                "cached_text": cached_text,
                                "cached_filename": current_pdf_name,
                                "research_mode": research_mode
                            },
                            timeout=120
                        )
                    else:
                        # New PDF - need to extract text
                        progress_placeholder.progress(0.2)
                        status_placeholder.info("📄 Extracting text from PDF...")
                        time.sleep(0.3)

                        # Upload PDF to API with form data
                        files = {"file": (uploaded_file.name, pdf_content, "application/pdf")}

                        progress_placeholder.progress(0.5)
                        status_placeholder.info(f"🤖 Processing PDF content ({mode_text})...")

                        response = get_api_session().post(
                            f"{API_BASE_URL}/process-pdf",
                            files=files,
                            data={
                                "summarization_mode": summarization_mode,
                                "output_length": output_length,
                                "research_mode": research_mode
                            },
                            timeout=120
                        )

                    progress_placeholder.progress(0.9)
                    status_placeholder.info("📝 Formatting structured notes...")
                    time.sleep(0.3)

                    progress_placeholder.progress(1.0)
                    progress_placeholder.empty()
                    status_placeholder.empty()

                    if response.status_code == 200:
                        result = response.json()

                        # Phase 5: Cache extracted text for future use (only from /process-pdf)
                        if not use_cached_text and result.get('extracted_text'):
                            st.session_state.last_pdf_name = current_pdf_name
                            st.session_state.last_pdf_size = current_pdf_size
                            st.session_state.last_pdf_text = result['extracted_text']

                        if result['success']:
                            st.success(f"✅ Notes generated from PDF: {uploaded_file.name}")

                            # Update LLM provider info in session state
                            try:
                                stats_resp = get_api_session().get(f"{API_BASE_URL}/stats", timeout=60)
                                if stats_resp.status_code == 200:
                                    stats = stats_resp.json()
                                    llm_info = stats.get('llm', {})
                                    st.session_state.last_llm_provider = llm_info.get('provider', 'unknown')
                                    st.session_state.last_llm_model = llm_info.get('model', 'unknown')
                            except:
                                pass  # Don't fail generation if stats fetch fails

                            # Save to history
                            note_entry = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'query': f"PDF: {uploaded_file.name}",
                                'type': 'pdf',
                                'notes': result['notes'],
                                'sources_used': result.get('sources_used', 0),
                                'from_kb': False
                            }
                            st.session_state.note_history.append(note_entry)
                            if len(st.session_state.note_history) > 6:
                                st.session_state.note_history.pop(0)

                            # Store for persistence across reruns
                            st.session_state.last_generated_notes = result['notes']
                            st.session_state.last_vision_data = result.get('vision_data')
                            st.session_state.notes_generated_time = datetime.now()
                            st.session_state.last_generation_metadata = {
                                'query': f"PDF: {uploaded_file.name}",
                                'type': 'pdf',
                                'sources_used': result.get('sources_used', 0),
                                'file_size': f"{file_size_mb:.2f} MB"
                            }
                            st.session_state.show_copy_box = False  # Reset copy box state
                            st.session_state.saved_to_kb_recent = False  # Reset save state for new notes
                            # Rerun to update sidebar history immediately
                            st.rerun()
                        else:
                            _err = result.get('error', 'Unknown error')
                            if _is_rate_limit_error(_err):
                                _show_rate_limit_dialog(_err, key_suffix="_pdf")
                            else:
                                st.error(f"❌ Error: {_err}")
                    else:
                        _err = f"{response.status_code} - {response.text}"
                        if _is_rate_limit_error(_err):
                            _show_rate_limit_dialog(_err, key_suffix="_pdf_api")
                        else:
                            st.error(f"❌ API Error: {_err}")

            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.empty()
                _err = str(e)
                if _is_rate_limit_error(_err):
                    _show_rate_limit_dialog(_err, key_suffix="_pdf_exc")
                else:
                    st.error(f"❌ Error processing PDF: {_err}")

        elif query_input:
            # Progress indicator container
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Step 1: Routing
                progress_placeholder.progress(0.2)
                status_placeholder.info("🔄 Detecting query type and routing...")
                time.sleep(0.3)

                # Step 2: Processing
                progress_placeholder.progress(0.4)
                _is_pdf_url = (
                    query_input.startswith(('http://', 'https://'))
                    and (query_input.lower().endswith('.pdf') or '/pdf/' in query_input.lower())
                )
                if _is_pdf_url:
                    status_placeholder.info("📥 Downloading and processing PDF from URL (this may take a moment)...")
                elif query_input.startswith(('http://', 'https://', 'www.')):
                    status_placeholder.info("🌐 Scraping web content...")
                elif len(query_input) > 500:
                    status_placeholder.info("📄 Processing text input...")
                elif search_mode == "both":
                    status_placeholder.info("🔗 Searching knowledge base and web for comprehensive coverage...")
                elif search_mode == "web_search":
                    status_placeholder.info("🌐 Searching the web with AI agent...")
                elif search_mode == "kb_only":
                    status_placeholder.info("📚 Searching knowledge base...")
                else:
                    status_placeholder.info("🔍 Searching knowledge base (web fallback enabled)...")

                # Make API request (PDF URLs need longer timeout: download + extraction + LLM)
                _req_timeout = 180 if _is_pdf_url else 120
                response = get_api_session().post(
                    f"{API_BASE_URL}/generate-notes",
                    json={
                        "query": query_input,
                        "summarization_mode": summarization_mode,
                        "summary_length": output_length,
                        "search_mode": search_mode,
                        "research_mode": research_mode
                    },
                    timeout=_req_timeout
                )

                # Step 3: Summarizing
                progress_placeholder.progress(0.7)
                mode_text_map = {
                    "paragraph_summary": "paragraph summary",
                    "important_points": "important points",
                    "key_highlights": "key highlights"
                }
                mode_text = mode_text_map.get(summarization_mode, "summary")
                status_placeholder.info(f"🤖 Generating {mode_text}...")
                time.sleep(0.3)

                # Step 4: Creating Notes
                progress_placeholder.progress(0.9)
                status_placeholder.info("📝 Formatting structured notes...")
                time.sleep(0.3)

                # Complete
                progress_placeholder.progress(1.0)
                status_placeholder.empty()
                progress_placeholder.empty()

                if response.status_code == 200:
                    result = response.json()

                    if result['success']:
                        st.success("✅ Notes generated successfully!")

                        # Update LLM provider info in session state
                        try:
                            stats_resp = get_api_session().get(f"{API_BASE_URL}/stats", timeout=60)
                            if stats_resp.status_code == 200:
                                stats = stats_resp.json()
                                llm_info = stats.get('llm', {})
                                st.session_state.last_llm_provider = llm_info.get('provider', 'unknown')
                                st.session_state.last_llm_model = llm_info.get('model', 'unknown')
                        except:
                            pass  # Don't fail generation if stats fetch fails

                        # Save to history (keep last 6)
                        note_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'query': query_input,
                            'type': result['query_type'],
                            'notes': result['notes'],
                            'sources_used': result['sources_used'],
                            'from_kb': result['from_kb']
                        }
                        st.session_state.note_history.append(note_entry)
                        if len(st.session_state.note_history) > 6:
                            st.session_state.note_history.pop(0)

                        # Store for persistence across reruns
                        st.session_state.last_generated_notes = result['notes']
                        st.session_state.last_vision_data = result.get('vision_data')
                        st.session_state.notes_generated_time = datetime.now()
                        st.session_state.last_generation_metadata = {
                            'query': query_input[:100] + '...' if len(query_input) > 100 else query_input,
                            'type': result['query_type'],
                            'sources_used': result['sources_used'],
                            'from_kb': result['from_kb'],
                            'search_mode': search_mode,
                            'message': result.get('message', '')
                        }
                        st.session_state.show_copy_box = False  # Reset copy box state
                        st.session_state.saved_to_kb_recent = False  # Reset save state for new notes
                        # Rerun to update sidebar history immediately
                        st.rerun()
                    else:
                        _err = result.get('error', 'Unknown error')
                        if _is_rate_limit_error(_err):
                            _show_rate_limit_dialog(_err, key_suffix="_query")
                        else:
                            st.error(f"❌ Error: {_err}")
                else:
                    _err = f"{response.status_code} - {response.text}"
                    if _is_rate_limit_error(_err):
                        _show_rate_limit_dialog(_err, key_suffix="_query_api")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")

            except requests.exceptions.Timeout:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.empty()
                _err = str(e)
                if _is_rate_limit_error(_err):
                    _show_rate_limit_dialog(_err, key_suffix="_query_exc")
                else:
                    st.error(f"❌ Error: {_err}")
        else:
            st.warning("⚠️ Please enter a query")

    # ==========================================================================
    # SECTION: Recently Generated Notes (Independent, Persistent)
    # ==========================================================================
    if st.session_state.get('last_generated_notes'):
        st.divider()

        # Header with close button
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.markdown("### 📚 Recently Generated Notes")
        with header_col2:
            if st.button("✕", key="close_generated_notes", help="Close this section"):
                st.session_state.last_generated_notes = None
                st.session_state.last_vision_data = None
                st.session_state.last_generation_metadata = None
                st.session_state.show_copy_box = False
                st.session_state.show_save_kb_recent = False
                st.session_state.saved_to_kb_recent = False
                st.session_state.edit_mode_recent = False
                st.session_state.edit_undo_recent = None
                st.session_state.edit_undo_stack_recent = []
                st.session_state.edit_redo_stack_recent = []
                st.session_state.edit_current_text_recent = None
                st.rerun()

        # Display metadata if available
        metadata = st.session_state.get('last_generation_metadata')
        if metadata:
            _meta_type = metadata.get('type', 'N/A')
            _is_topic_meta = _meta_type == 'topic'

            if _is_topic_meta:
                # Topic queries: show 4 columns including Search Mode
                col1, col2, col3, col4 = st.columns(4)
            else:
                # URL/Text/PDF queries: show 3 columns (no Search Mode)
                col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Query Type", _meta_type.upper())
            with col2:
                st.metric("Sources Used", metadata.get('sources_used', 0))
            with col3:
                _search_mode_meta = metadata.get('search_mode', 'auto')
                if _search_mode_meta == 'both':
                    source_label = "KB + Web"
                elif metadata.get('from_kb'):
                    source_label = "Knowledge Base"
                elif _meta_type == 'url':
                    source_label = "Web Scrape"
                elif _meta_type == 'text' or _meta_type == 'pdf':
                    source_label = "Direct Input"
                elif _search_mode_meta in ('web_search', 'auto'):
                    source_label = "Web Search"
                else:
                    source_label = "N/A"
                st.metric("Source", source_label)

            if _is_topic_meta:
                with col4:
                    mode_labels = {"auto": "Auto", "kb_only": "KB Only", "web_search": "Web Search", "both": "KB + Web"}
                    st.metric("Search Mode", mode_labels.get(_search_mode_meta, 'Auto'))

        # Display notes (view mode or edit mode)
        if st.session_state.edit_mode_recent:
            st.markdown("**✏️ Editing Notes** — make changes below, then Save or Cancel.")
            edited_recent = st.text_area(
                "Edit notes:",
                height=400,
                key="edit_textarea_recent",
                label_visibility="collapsed",
                on_change=_on_edit_change_recent
            )

            # Edit mode buttons: Save | ↩️ Undo | ↪️ Redo | Cancel
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                if st.button("💾 Save", use_container_width=True, key="edit_save_recent"):
                    final_text = st.session_state.get('edit_textarea_recent', st.session_state.last_generated_notes)
                    # Store current version for view-mode undo before overwriting
                    st.session_state.edit_undo_recent = st.session_state.last_generated_notes
                    st.session_state.last_generated_notes = final_text
                    # Update matching entry in note_history
                    meta = st.session_state.get('last_generation_metadata', {})
                    if meta and st.session_state.get('note_history'):
                        for entry in st.session_state.note_history:
                            if entry.get('timestamp') == meta.get('timestamp') and entry.get('query') == meta.get('query'):
                                entry['notes'] = final_text
                                break
                    # Cleanup edit stacks
                    st.session_state.edit_undo_stack_recent = []
                    st.session_state.edit_redo_stack_recent = []
                    st.session_state.edit_current_text_recent = None
                    st.session_state.edit_mode_recent = False
                    st.rerun()
            with ecol2:
                st.button("↩️", use_container_width=True, key="edit_undo_btn_recent",
                          disabled=(len(st.session_state.edit_undo_stack_recent) == 0),
                          help="Undo", on_click=_undo_edit_recent)
            with ecol3:
                st.button("↪️", use_container_width=True, key="edit_redo_btn_recent",
                          disabled=(len(st.session_state.edit_redo_stack_recent) == 0),
                          help="Redo", on_click=_redo_edit_recent)
            with ecol4:
                if st.button("✕ Cancel", use_container_width=True, key="edit_cancel_recent"):
                    st.session_state.edit_undo_stack_recent = []
                    st.session_state.edit_redo_stack_recent = []
                    st.session_state.edit_current_text_recent = None
                    st.session_state.edit_mode_recent = False
                    st.rerun()
        else:
            st.markdown('<div class="notes-container">', unsafe_allow_html=True)
            st.markdown(st.session_state.last_generated_notes)
            st.markdown('</div>', unsafe_allow_html=True)

            # Research Mode: Show figures & visual analysis in collapsible section
            vision_json = st.session_state.get('last_vision_data')
            if vision_json:
                import json as json_mod
                import base64 as b64_mod
                import re as re_mod
                try:
                    figures = json_mod.loads(vision_json)
                    if figures:
                        with st.expander(f"Figures & Visual Analysis ({len(figures)} pages)", expanded=False):
                            for idx, fig in enumerate(figures):
                                st.markdown(f"**Page {fig['page']}**")
                                if fig.get('image_b64'):
                                    img_bytes = b64_mod.b64decode(fig['image_b64'])
                                    st.image(img_bytes, width=500)
                                if fig.get('description'):
                                    desc = fig['description']
                                    # Clean any remaining HTML <br> tags
                                    desc = re_mod.sub(r'<br\s*/?>', '\n', desc)
                                    desc = re_mod.sub(r'<[^>]+>', '', desc)
                                    # Downshift headings: # → ####, ## → #####, ### → ######
                                    desc = re_mod.sub(
                                        r'^(#{1,3})\s',
                                        lambda m: '#' * min(len(m.group(1)) + 3, 6) + ' ',
                                        desc,
                                        flags=re_mod.MULTILINE
                                    )
                                    # Remove "No X present/found" lines and their headings
                                    lines = desc.split('\n')
                                    cleaned = []
                                    i = 0
                                    while i < len(lines):
                                        s = lines[i].strip()
                                        if re_mod.search(r'\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)', s, re_mod.IGNORECASE):
                                            if cleaned and re_mod.match(r'^#{1,6}\s', cleaned[-1].strip()):
                                                cleaned.pop()
                                            i += 1
                                            continue
                                        if re_mod.match(r'^#{1,6}\s', s) and i + 1 < len(lines):
                                            ns = lines[i + 1].strip()
                                            if re_mod.search(r'\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)', ns, re_mod.IGNORECASE):
                                                i += 2
                                                continue
                                        cleaned.append(lines[i])
                                        i += 1
                                    desc = '\n'.join(cleaned).strip()
                                    desc = re_mod.sub(r'\n{3,}', '\n\n', desc)
                                    if desc:
                                        st.markdown(desc)
                                if idx < len(figures) - 1:
                                    st.divider()
                except Exception:
                    pass

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.download_button(
                    label="📥 Download Notes",
                    data=st.session_state.last_generated_notes,
                    file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_generated_persistent"
                )
            with col2:
                if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_generated_persistent"):
                    st.session_state.show_copy_box = True
            with col3:
                if st.session_state.get('saved_to_kb_recent', False):
                    st.button("✅ Saved to KB", use_container_width=True, key="saved_to_kb_btn_recent", disabled=True)
                else:
                    if st.button("💾 Save to KB", use_container_width=True, key="save_to_kb_recent"):
                        if not st.session_state.show_save_kb_recent:
                            # Initialize form values when opening (prevents input loss on rerun)
                            meta = st.session_state.get('last_generation_metadata', {})
                            query = meta.get('query', '')
                            query_type = meta.get('type', 'topic')
                            st.session_state.save_kb_title_recent = query[:80] if len(query) <= 80 else query[:77] + "..."
                            st.session_state.save_kb_topic_recent = query if query_type == 'topic' and len(query) <= 50 else ""
                            if query_type == 'pdf':
                                st.session_state.save_kb_source_recent = f"Generated Notes - PDF: {query.replace('PDF: ', '')}" if query.startswith('PDF: ') else "Generated Notes - PDF"
                            elif query_type == 'url':
                                st.session_state.save_kb_source_recent = "Generated Notes - URL"
                            elif query_type == 'text':
                                st.session_state.save_kb_source_recent = "Generated Notes - Text Input"
                            else:
                                st.session_state.save_kb_source_recent = "Generated Notes - Topic Query"
                            st.session_state.save_kb_url_recent = query if query_type == 'url' else ""
                            st.session_state.save_kb_query_type_recent = query_type
                            # Check for PDF duplicate when form opens (cache result in session state)
                            st.session_state.save_kb_pdf_duplicate_recent = False
                            if query_type == 'pdf':
                                try:
                                    pdf_name = query.replace('PDF: ', '') if query.startswith('PDF: ') else ''
                                    if pdf_name:
                                        check_resp = get_api_session().post(
                                            f"{API_BASE_URL}/search-kb",
                                            json={"query": pdf_name, "k": 5, "threshold": 0.8},
                                            timeout=60
                                        )
                                        if check_resp.status_code == 200:
                                            for doc in check_resp.json().get('results', []):
                                                doc_source = doc.get('metadata', {}).get('source', '')
                                                if 'Generated Notes - PDF' in doc_source and pdf_name.lower() in doc_source.lower():
                                                    st.session_state.save_kb_pdf_duplicate_recent = True
                                                    break
                                except:
                                    pass
                        st.session_state.show_save_kb_recent = not st.session_state.show_save_kb_recent
                        st.rerun()
            with col4:
                if st.button("✏️ Edit", use_container_width=True, key="edit_recent"):
                    st.session_state.edit_mode_recent = True
                    st.session_state.edit_undo_stack_recent = []
                    st.session_state.edit_redo_stack_recent = []
                    st.session_state.edit_current_text_recent = st.session_state.last_generated_notes
                    st.session_state.edit_textarea_recent = st.session_state.last_generated_notes
                    st.rerun()

            # Undo Last Edit button (visible in view mode when undo is available)
            if st.session_state.edit_undo_recent is not None:
                if st.button("↩️ Undo Last Edit", key="undo_last_edit_recent"):
                    st.session_state.last_generated_notes = st.session_state.edit_undo_recent
                    # Update matching entry in note_history
                    meta = st.session_state.get('last_generation_metadata', {})
                    if meta and st.session_state.get('note_history'):
                        for entry in st.session_state.note_history:
                            if entry.get('timestamp') == meta.get('timestamp') and entry.get('query') == meta.get('query'):
                                entry['notes'] = st.session_state.edit_undo_recent
                                break
                    st.session_state.edit_undo_recent = None
                    st.rerun()

            # Dismissible success message after saving to KB
            if st.session_state.get('saved_to_kb_recent', False):
                msg_col1, msg_col2 = st.columns([11, 1])
                with msg_col1:
                    st.success("✅ Notes saved to Knowledge Base successfully!")
                with msg_col2:
                    if st.button("✕", key="close_save_success_recent", help="Dismiss"):
                        st.session_state.saved_to_kb_recent = False
                        st.rerun()

            if st.session_state.get('show_copy_box', False):
                st.markdown("**📋 Copy the text below:**")
                st.text_area(
                    "Select all (Ctrl+A) and copy (Ctrl+C):",
                    value=st.session_state.last_generated_notes,
                    height=200,
                    key="copy_text_area_persistent"
                )

        # Save to KB Form (shown when button clicked, hidden during edit mode)
        if st.session_state.get('show_save_kb_recent', False) and not st.session_state.edit_mode_recent:
            st.markdown("---")
            st.markdown("#### 💾 Save Notes to Knowledge Base")

            # Get stored query type for conditional URL field
            query_type = st.session_state.get('save_kb_query_type_recent', 'topic')

            # Form fields (values stored in session state, no value= parameter to prevent input loss)
            save_title = st.text_input("Title *", key="save_kb_title_recent",
                                       help="A descriptive title for these notes")
            save_topic = st.text_input("Topic *", key="save_kb_topic_recent",
                                       placeholder="Add the topic name",
                                       help="Category/topic for organizing in KB")
            save_source = st.text_input("Source", key="save_kb_source_recent",
                                        help="Source of the content (auto-filled)")
            if query_type == 'url':
                save_url = st.text_input("URL", key="save_kb_url_recent")
            else:
                save_url = st.session_state.get('save_kb_url_recent', '')

            # PDF duplicate warning (uses cached result from when form opened)
            if st.session_state.get('save_kb_pdf_duplicate_recent', False):
                st.warning("📋 It looks like notes from this PDF have already been saved to the Knowledge Base. You may proceed if you wish to save another copy.")

            # Save/Cancel buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Save", use_container_width=True, key="confirm_save_kb_recent"):
                    if not save_title.strip():
                        st.error("Please enter a title")
                    elif not save_topic.strip():
                        st.error("Please enter a topic")
                    else:
                        # Make API call to save
                        try:
                            response = get_api_session().post(
                                f"{API_BASE_URL}/update-kb",
                                json={
                                    "documents": [{
                                        "content": st.session_state.last_generated_notes,
                                        "title": save_title.strip(),
                                        "topic": save_topic.strip(),
                                        "source": save_source.strip(),
                                        "url": save_url.strip() if save_url else ""
                                    }]
                                },
                                timeout=60
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success'):
                                    st.session_state.show_save_kb_recent = False
                                    st.session_state.saved_to_kb_recent = True
                                    # Clear topics cache to refresh Quick Topics
                                    fetch_topics.clear()
                                    st.rerun()
                                else:
                                    st.error(f"Failed to save: {result.get('error', 'Unknown error')}")
                            else:
                                st.error(f"API Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error saving to KB: {str(e)}")
            with btn_col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_save_kb_recent"):
                    st.session_state.show_save_kb_recent = False
                    st.rerun()

    # ==========================================================================
    # SECTION: History Note Viewer (Independent)
    # ==========================================================================
    if st.session_state.get('selected_history'):
        st.divider()
        hist = st.session_state.selected_history

        # Header with close button - prominent styling
        st.markdown("---")
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.markdown("### 📜 History Note Viewer")
            st.info(f"Viewing: **{hist['query'][:50]}{'...' if len(hist['query']) > 50 else ''}** ({hist['timestamp']})")
        with header_col2:
            if st.button("✕", key="close_history_view", help="Close history view"):
                st.session_state.selected_history = None
                st.session_state.show_copy_box_history = False
                st.session_state.show_save_kb_history = False
                st.session_state.saved_to_kb_history = False
                st.session_state.edit_mode_history = False
                st.session_state.edit_undo_history = None
                st.session_state.edit_undo_stack_history = []
                st.session_state.edit_redo_stack_history = []
                st.session_state.edit_current_text_history = None
                st.rerun()

        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Type", hist['type'].upper())
        with col2:
            st.metric("Sources Used", hist['sources_used'])
        with col3:
            st.metric("From KB", "Yes" if hist['from_kb'] else "No")

        # Display notes (view mode or edit mode)
        if st.session_state.edit_mode_history:
            st.markdown("**✏️ Editing Notes** — make changes below, then Save or Cancel.")
            edited_history = st.text_area(
                "Edit notes:",
                height=400,
                key="edit_textarea_history",
                label_visibility="collapsed",
                on_change=_on_edit_change_history
            )

            # Edit mode buttons: Save | ↩️ Undo | ↪️ Redo | Cancel
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                if st.button("💾 Save", use_container_width=True, key="edit_save_history"):
                    final_text = st.session_state.get('edit_textarea_history', hist['notes'])
                    # Store current version for view-mode undo before overwriting
                    st.session_state.edit_undo_history = hist['notes']
                    # Update selected_history
                    st.session_state.selected_history['notes'] = final_text
                    # Update matching entry in note_history
                    if st.session_state.get('note_history'):
                        for entry in st.session_state.note_history:
                            if entry.get('timestamp') == hist.get('timestamp') and entry.get('query') == hist.get('query'):
                                entry['notes'] = final_text
                                break
                    # Cleanup edit stacks
                    st.session_state.edit_undo_stack_history = []
                    st.session_state.edit_redo_stack_history = []
                    st.session_state.edit_current_text_history = None
                    st.session_state.edit_mode_history = False
                    st.rerun()
            with ecol2:
                st.button("↩️", use_container_width=True, key="edit_undo_btn_history",
                          disabled=(len(st.session_state.edit_undo_stack_history) == 0),
                          help="Undo", on_click=_undo_edit_history)
            with ecol3:
                st.button("↪️", use_container_width=True, key="edit_redo_btn_history",
                          disabled=(len(st.session_state.edit_redo_stack_history) == 0),
                          help="Redo", on_click=_redo_edit_history)
            with ecol4:
                if st.button("✕ Cancel", use_container_width=True, key="edit_cancel_history"):
                    st.session_state.edit_undo_stack_history = []
                    st.session_state.edit_redo_stack_history = []
                    st.session_state.edit_current_text_history = None
                    st.session_state.edit_mode_history = False
                    st.rerun()
        else:
            st.markdown('<div class="notes-container">', unsafe_allow_html=True)
            st.markdown(hist['notes'])
            st.markdown('</div>', unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.download_button(
                    label="📥 Download",
                    data=hist['notes'],
                    file_name=f"notes_history_{hist['timestamp'].replace(':', '-')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_history"
                )
            with col2:
                if st.button("📋 Copy", use_container_width=True, key="copy_history"):
                    st.session_state.show_copy_box_history = True
            with col3:
                if st.session_state.get('saved_to_kb_history', False):
                    st.button("✅ Saved to KB", use_container_width=True, key="saved_to_kb_btn_history", disabled=True)
                else:
                    if st.button("💾 Save to KB", use_container_width=True, key="save_to_kb_history"):
                        if not st.session_state.show_save_kb_history:
                            # Initialize form values when opening (prevents input loss on rerun)
                            query = hist.get('query', '')
                            query_type = hist.get('type', 'topic')
                            st.session_state.save_kb_title_history = query[:80] if len(query) <= 80 else query[:77] + "..."
                            st.session_state.save_kb_topic_history = query if query_type == 'topic' and len(query) <= 50 else ""
                            if query_type == 'pdf':
                                st.session_state.save_kb_source_history = f"Generated Notes - PDF: {query.replace('PDF: ', '')}" if query.startswith('PDF: ') else "Generated Notes - PDF"
                            elif query_type == 'url':
                                st.session_state.save_kb_source_history = "Generated Notes - URL"
                            elif query_type == 'text':
                                st.session_state.save_kb_source_history = "Generated Notes - Text Input"
                            else:
                                st.session_state.save_kb_source_history = "Generated Notes - Topic Query"
                            st.session_state.save_kb_url_history = query if query_type == 'url' else ""
                            st.session_state.save_kb_query_type_history = query_type
                            # Check for PDF duplicate when form opens (cache result in session state)
                            st.session_state.save_kb_pdf_duplicate_history = False
                            if query_type == 'pdf':
                                try:
                                    pdf_name = query.replace('PDF: ', '') if query.startswith('PDF: ') else ''
                                    if pdf_name:
                                        check_resp = get_api_session().post(
                                            f"{API_BASE_URL}/search-kb",
                                            json={"query": pdf_name, "k": 5, "threshold": 0.8},
                                            timeout=60
                                        )
                                        if check_resp.status_code == 200:
                                            for doc in check_resp.json().get('results', []):
                                                doc_source = doc.get('metadata', {}).get('source', '')
                                                if 'Generated Notes - PDF' in doc_source and pdf_name.lower() in doc_source.lower():
                                                    st.session_state.save_kb_pdf_duplicate_history = True
                                                    break
                                except:
                                    pass
                        st.session_state.show_save_kb_history = not st.session_state.show_save_kb_history
                        st.rerun()
            with col4:
                if st.button("✏️ Edit", use_container_width=True, key="edit_history"):
                    st.session_state.edit_mode_history = True
                    st.session_state.edit_undo_stack_history = []
                    st.session_state.edit_redo_stack_history = []
                    st.session_state.edit_current_text_history = hist['notes']
                    st.session_state.edit_textarea_history = hist['notes']
                    st.rerun()

            # Undo Last Edit button (visible in view mode when undo is available)
            if st.session_state.edit_undo_history is not None:
                if st.button("↩️ Undo Last Edit", key="undo_last_edit_history"):
                    restored = st.session_state.edit_undo_history
                    st.session_state.selected_history['notes'] = restored
                    if st.session_state.get('note_history'):
                        for entry in st.session_state.note_history:
                            if entry.get('timestamp') == hist.get('timestamp') and entry.get('query') == hist.get('query'):
                                entry['notes'] = restored
                                break
                    st.session_state.edit_undo_history = None
                    st.rerun()

            # Dismissible success message after saving to KB
            if st.session_state.get('saved_to_kb_history', False):
                msg_col1, msg_col2 = st.columns([11, 1])
                with msg_col1:
                    st.success("✅ Notes saved to Knowledge Base successfully!")
                with msg_col2:
                    if st.button("✕", key="close_save_success_history", help="Dismiss"):
                        st.session_state.saved_to_kb_history = False
                        st.rerun()

            if st.session_state.get('show_copy_box_history', False):
                st.text_area(
                    "Copy this text:",
                    value=hist['notes'],
                    height=150,
                    key="copy_history_text"
                )

        # Save to KB Form for History (shown when button clicked, hidden during edit mode)
        if st.session_state.get('show_save_kb_history', False) and not st.session_state.edit_mode_history:
            st.markdown("---")
            st.markdown("#### 💾 Save Notes to Knowledge Base")

            # Get stored query type for conditional URL field
            query_type = st.session_state.get('save_kb_query_type_history', 'topic')

            # Form fields (values stored in session state, no value= parameter to prevent input loss)
            save_title_h = st.text_input("Title *", key="save_kb_title_history",
                                         help="A descriptive title for these notes")
            save_topic_h = st.text_input("Topic *", key="save_kb_topic_history",
                                         placeholder="Add the topic name",
                                         help="Category/topic for organizing in KB")
            save_source_h = st.text_input("Source", key="save_kb_source_history",
                                          help="Source of the content (auto-filled)")
            if query_type == 'url':
                save_url_h = st.text_input("URL", key="save_kb_url_history")
            else:
                save_url_h = st.session_state.get('save_kb_url_history', '')

            # PDF duplicate warning (uses cached result from when form opened)
            if st.session_state.get('save_kb_pdf_duplicate_history', False):
                st.warning("📋 It looks like notes from this PDF have already been saved to the Knowledge Base. You may proceed if you wish to save another copy.")

            # Save/Cancel buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Save", use_container_width=True, key="confirm_save_kb_history"):
                    if not save_title_h.strip():
                        st.error("Please enter a title")
                    elif not save_topic_h.strip():
                        st.error("Please enter a topic")
                    else:
                        # Make API call to save
                        try:
                            response = get_api_session().post(
                                f"{API_BASE_URL}/update-kb",
                                json={
                                    "documents": [{
                                        "content": hist['notes'],
                                        "title": save_title_h.strip(),
                                        "topic": save_topic_h.strip(),
                                        "source": save_source_h.strip(),
                                        "url": save_url_h.strip() if save_url_h else ""
                                    }]
                                },
                                timeout=60
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success'):
                                    st.session_state.show_save_kb_history = False
                                    st.session_state.saved_to_kb_history = True
                                    # Clear topics cache to refresh Quick Topics
                                    fetch_topics.clear()
                                    st.rerun()
                                else:
                                    st.error(f"Failed to save: {result.get('error', 'Unknown error')}")
                            else:
                                st.error(f"API Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error saving to KB: {str(e)}")
            with btn_col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_save_kb_history"):
                    st.session_state.show_save_kb_history = False
                    st.rerun()

# =============================================================================
# AI CHAT TAB
# =============================================================================
with tab_chat:
    # ---- Session state initialization ----
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "chat_provider_info" not in st.session_state:
        st.session_state.chat_provider_info = ""
    if "research_state" not in st.session_state:
        st.session_state.research_state = None
    if "editing_msg_idx" not in st.session_state:
        st.session_state.editing_msg_idx = None

    CHAT_API = f"{API_BASE_URL}/chat"
    CHAT_INPUT_MAX_CHARS = 50000

    # ---- Mode selector row ----
    chat_col1, chat_col2, chat_col3 = st.columns([2, 1, 1])
    with chat_col1:
        chat_mode_labels = {
            "chat": "Chat",
            "answer_writer": "Exam Answer",
            "explain": "Explain Concept",
            "compare": "Compare Topics",
            "socratic": "Socratic Tutor",
            "paper_analysis": "Paper Analysis",
            "research": "Research Writer",
        }
        chat_mode_descriptions = {
            "chat": "Ask anything — your intelligent study companion",
            "answer_writer": "Model answers for exam preparation",
            "explain": "Concept explanations tailored to your level",
            "compare": "Structured comparison between two topics",
            "socratic": "Guided learning through questions",
            "paper_analysis": "Analyze and discuss research papers",
            "research": "Assisted research paper writing",
        }
        chat_mode = st.selectbox(
            "Mode",
            options=list(chat_mode_labels.keys()),
            format_func=lambda x: chat_mode_labels[x],
            key="chat_mode_select",
        )
        st.caption(f"ℹ️ {chat_mode_descriptions[chat_mode]}")
    with chat_col2:
        chat_use_kb = st.toggle("Auto Search", value=True, key="chat_kb_toggle", help="ON: Searches Knowledge Base and web for context before answering. OFF: AI answers purely from its own knowledge.")
    with chat_col3:
        if st.button("New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chat_session_id = None
            st.session_state.chat_provider_info = ""
            st.session_state.research_state = None
            st.session_state.editing_msg_idx = None
            st.rerun()

    # ---- Mode-specific extra controls ----
    extra_params = {}
    if chat_mode == "explain":
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            extra_params["explain_level"] = st.selectbox(
                "Level", ["eli5", "beginner", "intermediate", "advanced"],
                format_func=lambda x: {"eli5": "ELI5 (Super Simple)", "beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"}[x],
                index=2, key="explain_level"
            )
        with ex_col2:
            extra_params["explain_style"] = st.selectbox(
                "Style", ["technical", "analogy", "visual"],
                format_func=lambda x: {"technical": "Technical", "analogy": "Using Analogies", "visual": "Visual (Diagrams & Tables)"}[x],
                key="explain_style"
            )
        with ex_col3:
            if extra_params.get("explain_style") == "analogy":
                extra_params["analogy_domain"] = st.text_input(
                    "Analogy domain", value="everyday life", key="analogy_domain"
                )
    elif chat_mode == "answer_writer":
        extra_params["answer_depth"] = st.selectbox(
            "Answer length", ["brief", "standard", "detailed"],
            format_func=lambda x: {"brief": "Brief (100-150 words)", "standard": "Standard (400-600 words)", "detailed": "Detailed (800-1500 words)"}[x],
            index=1, key="answer_depth"
        )
    elif chat_mode == "compare":
        extra_params["compare_concept_2"] = st.text_input(
            "Second topic", key="compare_concept_2",
        )

    # ---- Helper to send a chat message to the API ----
    def _send_chat_message(user_message):
        """Send a message to the chat API and update session state."""
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]
        payload = {
            "message": user_message,
            "mode": chat_mode,
            "use_kb": chat_use_kb,
            "history": history,
            "session_id": st.session_state.chat_session_id,
            **extra_params,
        }
        try:
            session = get_api_session()
            resp = session.post(f"{CHAT_API}/message", json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    st.session_state.chat_session_id = data.get("session_id")
                    st.session_state.chat_provider_info = data.get("provider_used", "")
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": user_message}
                    )
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": data["response"],
                        "suggestions": data.get("suggestions", []),
                        "sources": data.get("sources", []),
                    })
                    return True
                else:
                    st.error(f"Chat error: {data.get('error', 'Unknown error')}")
            else:
                st.error(f"API returned status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the server is running.")
        except Exception as e:
            st.error(f"Error: {e}")
        return False

    # ==================================================================
    # RESEARCH PAPER MODE
    # ==================================================================
    if chat_mode == "research":
        st.markdown("#### Research Writer")

        rs = st.session_state.research_state

        # ---- Stage: Not started ----
        if rs is None:
            with st.form("research_start_form"):
                r_topic = st.text_input("Paper title / topic", key="r_topic")
                r_context = st.text_area(
                    "Initial context (what you know so far)",
                    height=150, key="r_context",
                    placeholder="Describe your research question, methodology, data, results...",
                )
                r_type = st.selectbox(
                    "Paper type", ["research", "survey", "case_study"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="r_type",
                )
                r_submit = st.form_submit_button("Start Analysis", type="primary")

            if r_submit and r_topic:
                with st.spinner("Analyzing your topic..."):
                    try:
                        session = get_api_session()
                        resp = session.post(
                            f"{CHAT_API}/research/start",
                            json={"topic": r_topic, "initial_context": r_context, "paper_type": r_type},
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("success"):
                                # Build initial AI message from analysis + questions
                                _init_parts = []
                                if data.get("analysis"):
                                    _init_parts.append(data["analysis"])
                                if data.get("info_available"):
                                    _init_parts.append(f"\n**Available information:** {', '.join(data['info_available'])}")
                                if data.get("info_missing"):
                                    _init_parts.append(f"\n**Still needed:** {', '.join(data['info_missing'])}")
                                _init_questions = data.get("questions", [])
                                if _init_questions:
                                    _init_parts.append("\n**Please provide the following:**")
                                    for _qi, _q in enumerate(_init_questions, 1):
                                        _init_parts.append(f"{_qi}. {_q}")
                                st.session_state.research_state = {
                                    "session_id": data["session_id"],
                                    "topic": r_topic,
                                    "paper_type": r_type,
                                    "stage": data.get("stage", "gathering"),
                                    "questions": _init_questions,
                                    "analysis": data.get("analysis", ""),
                                    "sufficiency": data.get("sufficiency_score", 0.0),
                                    "info_available": data.get("info_available", []),
                                    "info_missing": data.get("info_missing", []),
                                    "outline": data.get("outline", []),
                                    "sections": {},
                                    "sections_order": [],
                                    "full_paper": None,
                                    "related_papers": data.get("related_papers", []),
                                    "gathering_messages": [{"role": "assistant", "content": "\n".join(_init_parts)}] if _init_parts else [],
                                    "response_count": 0,
                                }
                                st.rerun()
                            else:
                                st.error(data.get("error", "Failed to start analysis"))
                        else:
                            st.error(f"API returned status {resp.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Make sure the server is running.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ---- Stage: Gathering (conversational) ----
        elif rs.get("stage") == "gathering":
            st.markdown(f"**Topic:** {rs['topic']}")
            score = rs.get("sufficiency", 0)
            st.progress(score, text=f"Information sufficiency: {score:.0%}")

            # Initialize gathering messages if missing (backward compat)
            if "gathering_messages" not in rs:
                _parts = []
                if rs.get("analysis"):
                    _parts.append(rs["analysis"])
                if rs.get("info_missing"):
                    _parts.append(f"\n**Still needed:** {', '.join(rs['info_missing'])}")
                for _qi, _q in enumerate(rs.get("questions", []), 1):
                    _parts.append(f"{_qi}. {_q}")
                rs["gathering_messages"] = [{"role": "assistant", "content": "\n".join(_parts)}] if _parts else []
                rs["response_count"] = 0

            # Display conversation
            gather_container = st.container(height=320)
            with gather_container:
                for _gm in rs["gathering_messages"]:
                    with st.chat_message(_gm["role"]):
                        st.markdown(_gm["content"])

            # User input
            _r_input = st.text_area(
                "Your response",
                placeholder="Provide the requested information here — you can address multiple questions at once...",
                key=f"research_gather_input_{rs.get('response_count', 0)}",
                height=100,
                label_visibility="collapsed",
            )

            _gc1, _gc2 = st.columns([3, 1])
            with _gc1:
                if st.button("Send", type="primary", key="research_gather_send", use_container_width=True):
                    if _r_input and _r_input.strip():
                        # Add user message to conversation
                        rs["gathering_messages"].append({"role": "user", "content": _r_input.strip()})
                        rs["response_count"] = rs.get("response_count", 0) + 1

                        with st.spinner("Analyzing your input..."):
                            try:
                                session = get_api_session()
                                resp = session.post(
                                    f"{CHAT_API}/research/continue",
                                    json={
                                        "session_id": rs["session_id"],
                                        "answers": {f"response_{rs['response_count']}": _r_input.strip()},
                                    },
                                    timeout=120,
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    rs["stage"] = data.get("stage", "gathering")
                                    rs["sufficiency"] = data.get("sufficiency_score", 0.0)
                                    rs["info_available"] = data.get("info_available", [])
                                    rs["info_missing"] = data.get("info_missing", [])
                                    rs["questions"] = data.get("questions", [])
                                    rs["analysis"] = data.get("analysis", "")
                                    if data.get("outline"):
                                        rs["outline"] = data["outline"]

                                    # Add new AI response to conversation
                                    if rs["stage"] == "gathering":
                                        _new_parts = []
                                        if data.get("analysis"):
                                            _new_parts.append(data["analysis"])
                                        if data.get("info_available"):
                                            _new_parts.append(f"\n**Available:** {', '.join(data['info_available'])}")
                                        if data.get("info_missing"):
                                            _new_parts.append(f"\n**Still needed:** {', '.join(data['info_missing'])}")
                                        _new_qs = data.get("questions", [])
                                        if _new_qs:
                                            _new_parts.append("\n**Next, please provide:**")
                                            for _qi, _q in enumerate(_new_qs, 1):
                                                _new_parts.append(f"{_qi}. {_q}")
                                        if _new_parts:
                                            rs["gathering_messages"].append({"role": "assistant", "content": "\n".join(_new_parts)})
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Please enter a response.")
            with _gc2:
                if st.button("Reset", key="reset_research", use_container_width=True):
                    st.session_state.research_state = None
                    st.rerun()

            # Related papers (only if relevant results exist)
            if rs.get("related_papers"):
                with st.expander("Related Academic Papers", expanded=False):
                    for p in rs["related_papers"][:5]:
                        title = p.get("title", "Untitled")
                        year = p.get("year", "")
                        url = p.get("url", "")
                        if url:
                            st.markdown(f"- **[{title}]({url})** ({year})")
                        else:
                            st.markdown(f"- **{title}** ({year})")

        # ---- Stage: Outlining ----
        elif rs.get("stage") == "outlining":
            st.markdown(f"**Topic:** {rs['topic']}")
            st.progress(1.0, text="Information sufficient!")
            st.markdown("#### Proposed Outline")
            outline = rs.get("outline", [])
            for item in outline:
                st.markdown(f"- {item}")

            ol_col1, ol_col2, ol_col3 = st.columns(3)
            with ol_col1:
                if st.button("Approve & Start Writing", type="primary", key="approve_outline"):
                    rs["stage"] = "writing"
                    rs["sections_order"] = [
                        item for item in outline
                        if not item.strip().lower().startswith(("abstract", "0."))
                    ]
                    st.rerun()
            with ol_col2:
                if st.button("Reset Research", key="reset_research_outline"):
                    st.session_state.research_state = None
                    st.rerun()

        # ---- Stage: Writing ----
        elif rs.get("stage") == "writing":
            st.markdown(f"**Topic:** {rs['topic']}")
            sections_order = rs.get("sections_order", [])
            completed = list(rs.get("sections", {}).keys())
            total = len(sections_order) if sections_order else 1
            done = len(completed)
            st.progress(done / total if total else 0, text=f"Writing: {done}/{total} sections")

            # Show completed sections
            for sec_name in completed:
                with st.expander(f"Section: {sec_name}", expanded=False):
                    st.markdown(rs["sections"][sec_name])

            # Next section to generate
            remaining = [s for s in sections_order if s not in completed]
            if remaining:
                next_sec = remaining[0]
                st.markdown(f"**Next:** {next_sec}")
                extra_instr = st.text_input(
                    "Additional instructions (optional)", key="section_extra_instr"
                )
                if st.button(f"Generate: {next_sec}", type="primary", key="gen_section"):
                    with st.spinner(f"Writing '{next_sec}'..."):
                        try:
                            session = get_api_session()
                            resp = session.post(
                                f"{CHAT_API}/research/generate-section",
                                json={
                                    "session_id": rs["session_id"],
                                    "section_name": next_sec,
                                    "additional_instructions": extra_instr or None,
                                },
                                timeout=180,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                if data.get("content"):
                                    rs["sections"][next_sec] = data["content"]
                                    st.rerun()
                                else:
                                    st.error(data.get("error", "No content generated"))
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.success("All sections generated!")
                if st.button("Assemble Final Paper", type="primary", key="assemble_paper"):
                    with st.spinner("Assembling paper and generating abstract..."):
                        try:
                            session = get_api_session()
                            resp = session.post(
                                f"{CHAT_API}/research/assemble",
                                json={"session_id": rs["session_id"], "generate_abstract": True},
                                timeout=180,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                if data.get("full_paper"):
                                    rs["stage"] = "complete"
                                    rs["full_paper"] = data["full_paper"]
                                    rs["word_count"] = data.get("word_count", 0)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            if st.button("Reset Research", key="reset_research_writing"):
                st.session_state.research_state = None
                st.rerun()

        # ---- Stage: Complete ----
        elif rs.get("stage") == "complete":
            st.markdown(f"**Topic:** {rs['topic']}")
            st.success(f"Paper complete! Word count: {rs.get('word_count', 'N/A')}")
            st.markdown(rs.get("full_paper", ""))
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    "Download .md",
                    data=rs.get("full_paper", ""),
                    file_name=f"{rs['topic'][:40].replace(' ', '_')}.md",
                    mime="text/markdown",
                    key="dl_research_paper",
                )
            with dl_col2:
                if st.button("New Research Paper", key="new_research"):
                    st.session_state.research_state = None
                    st.rerun()

    # ==================================================================
    # REGULAR CHAT MODES (not research)
    # ==================================================================
    else:
        # ---- Paper Analysis: text area for pasting paper ----
        if chat_mode == "paper_analysis" and not st.session_state.chat_messages:
            paper_text = st.text_area(
                "Paper content",
                height=180,
                placeholder="Paste the research paper content here...",
                key="paper_content_input",
                label_visibility="collapsed",
            )
            if st.button("Analyze Paper", type="primary", key="analyze_paper_btn", use_container_width=True):
                if paper_text and len(paper_text.strip()) > 50:
                    with st.spinner("Analyzing paper..."):
                        _send_chat_message(paper_text.strip())
                    st.rerun()
                else:
                    st.warning("Please provide the paper content to analyze.")

        # ---- CSS for resizable text area and chat container ----
        st.markdown("""<style>
            div[data-testid="stTextArea"] textarea {
                resize: vertical !important;
                min-height: 60px;
                max-height: 300px;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="height"] {
                resize: vertical !important;
                overflow: auto !important;
                min-height: 200px !important;
                max-height: 90vh !important;
            }
        </style>""", unsafe_allow_html=True)

        # ---- Chat message display ----
        chat_container = st.container(height=700)
        with chat_container:
            if not st.session_state.chat_messages:
                mode_hints = {
                    "chat": "What would you like to learn today?",
                    "answer_writer": "Enter any exam question to get a model answer.",
                    "explain": "Which concept would you like explained?",
                    "compare": "Enter the first topic below, and the second above.",
                    "socratic": "Enter a topic to begin guided learning.",
                    "paper_analysis": "Add your paper content above to begin.",
                }
                hint = mode_hints.get(chat_mode, "")
                st.markdown(
                    f"<div style='text-align:center; color:#888; padding:3rem 1rem;'>"
                    f"<p style='font-size:0.85rem;'>{hint}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                for idx, msg in enumerate(st.session_state.chat_messages):
                    with st.chat_message(msg["role"]):
                        # ---- Editing mode for this message ----
                        if st.session_state.editing_msg_idx == idx:
                            edited_text = st.text_area(
                                "Edit message",
                                value=msg["content"],
                                key=f"edit_area_{idx}",
                                height=150,
                                label_visibility="collapsed",
                            )
                            _ec1, _ec2, _ec3 = st.columns(3)
                            with _ec1:
                                if st.button("Save", key=f"save_edit_{idx}", type="primary", use_container_width=True):
                                    st.session_state.chat_messages[idx]["content"] = edited_text
                                    if msg["role"] == "user":
                                        # Truncate conversation and regenerate from this point
                                        st.session_state.chat_messages = st.session_state.chat_messages[:idx]
                                        st.session_state.editing_msg_idx = None
                                        with st.spinner("Regenerating..."):
                                            _send_chat_message(edited_text.strip())
                                        st.rerun()
                                    else:
                                        st.session_state.editing_msg_idx = None
                                        st.rerun()
                            with _ec2:
                                if st.button("Cancel", key=f"cancel_edit_{idx}", use_container_width=True):
                                    st.session_state.editing_msg_idx = None
                                    st.rerun()
                            with _ec3:
                                if st.button("Delete", key=f"delete_msg_{idx}", use_container_width=True):
                                    # Delete this message and its pair (user+assistant or just assistant)
                                    if msg["role"] == "user" and idx + 1 < len(st.session_state.chat_messages):
                                        st.session_state.chat_messages.pop(idx + 1)
                                        st.session_state.chat_messages.pop(idx)
                                    elif msg["role"] == "assistant" and idx > 0:
                                        st.session_state.chat_messages.pop(idx)
                                        st.session_state.chat_messages.pop(idx - 1)
                                    else:
                                        st.session_state.chat_messages.pop(idx)
                                    st.session_state.editing_msg_idx = None
                                    st.rerun()
                        else:
                            # ---- Normal display mode ----
                            content = msg["content"]
                            if msg["role"] == "user" and len(content) > 600:
                                st.markdown(content[:500] + "...")
                                with st.expander("Show full message"):
                                    st.markdown(content)
                            else:
                                st.markdown(content)

                            # Source badges for assistant messages
                            if msg["role"] == "assistant" and msg.get("sources"):
                                src_names = [s.get("title", "KB") for s in msg["sources"]]
                                src_types = set(s.get("type", "kb") for s in msg["sources"])
                                src_label = "Web" if "web" in src_types else "KB"
                                st.caption(f"{src_label} Sources: {', '.join(src_names[:3])}")

                            # Action buttons row
                            _a1, _a2, _a3 = st.columns([1, 1, 6])
                            with _a1:
                                if st.button("Edit", key=f"edit_btn_{idx}", use_container_width=True):
                                    st.session_state.editing_msg_idx = idx
                                    st.rerun()
                            with _a2:
                                if msg["role"] == "user":
                                    if st.button("Regen", key=f"regen_btn_{idx}", use_container_width=True):
                                        # Re-send same message, remove current response
                                        regen_text = msg["content"]
                                        st.session_state.chat_messages = st.session_state.chat_messages[:idx]
                                        with st.spinner("Regenerating..."):
                                            _send_chat_message(regen_text)
                                        st.rerun()

        # ---- Follow-up suggestion buttons ----
        if st.session_state.chat_messages:
            last_msg = st.session_state.chat_messages[-1]
            suggestions = last_msg.get("suggestions", []) if last_msg["role"] == "assistant" else []
            if suggestions:
                st.markdown(
                    "<p style='font-size:0.8rem; color:#888; margin-bottom:0.3rem;'>Suggested follow-ups:</p>",
                    unsafe_allow_html=True,
                )
                sug_cols = st.columns(min(len(suggestions), 3))
                for i, sug in enumerate(suggestions[:3]):
                    with sug_cols[i]:
                        if st.button(sug, key=f"sug_{i}_{len(st.session_state.chat_messages)}", use_container_width=True):
                            _send_chat_message(sug)
                            st.rerun()

        # ---- Input area (multi-line text area) ----
        chat_input = st.text_area(
            "Message",
            placeholder="Type your message...",
            key=f"chat_input_{len(st.session_state.chat_messages)}",
            label_visibility="collapsed",
            height=80,
        )
        # Character count and warning
        if chat_input:
            _char_count = len(chat_input)
            if _char_count > CHAT_INPUT_MAX_CHARS:
                st.error(f"Input too long ({_char_count:,} / {CHAT_INPUT_MAX_CHARS:,} chars). Please shorten your message.")
            elif _char_count > CHAT_INPUT_MAX_CHARS * 0.8:
                st.warning(f"Approaching limit: {_char_count:,} / {CHAT_INPUT_MAX_CHARS:,} characters")

        send_clicked = st.button("Send", type="primary", use_container_width=True, key="chat_send")

        if send_clicked and chat_input and chat_input.strip():
            if len(chat_input) > CHAT_INPUT_MAX_CHARS:
                st.error("Message is too long. Please shorten it before sending.")
            elif chat_mode == "compare" and not extra_params.get("compare_concept_2", "").strip():
                st.warning("Please enter the second topic to compare against.")
            else:
                with st.spinner("Thinking..."):
                    if _send_chat_message(chat_input.strip()):
                        st.rerun()

        # ---- Provider info and session actions ----
        if st.session_state.chat_messages:
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            with action_col1:
                if st.session_state.chat_provider_info:
                    st.caption(f"Provider: {st.session_state.chat_provider_info}")
            with action_col2:
                if st.session_state.chat_session_id:
                    if st.button("Export .md", key="prep_export", use_container_width=True):
                        try:
                            session_api = get_api_session()
                            export_resp = session_api.post(
                                f"{CHAT_API}/sessions/{st.session_state.chat_session_id}/export",
                                timeout=60,
                            )
                            if export_resp.status_code == 200:
                                st.session_state.chat_export_data = export_resp.json()
                        except Exception:
                            st.error("Export failed")
            with action_col3:
                if st.session_state.get("chat_export_data"):
                    st.download_button(
                        "Download",
                        data=st.session_state.chat_export_data.get("markdown", ""),
                        file_name=st.session_state.chat_export_data.get("filename", "chat.md"),
                        mime="text/markdown",
                        key="dl_export_chat",
                    )

        # ---- Session history (collapsible) ----
        with st.expander("Chat History", expanded=False):
            try:
                session_api = get_api_session()
                sessions_resp = session_api.get(f"{CHAT_API}/sessions", timeout=60)
                if sessions_resp.status_code == 200:
                    saved_sessions = sessions_resp.json().get("sessions", [])
                    if saved_sessions:
                        history_search = st.text_input(
                            "Search conversations",
                            placeholder="Search by keyword...",
                            key="chat_history_search",
                            label_visibility="collapsed",
                        )
                        if history_search and history_search.strip():
                            search_words = history_search.strip().lower().split()
                            filtered_sessions = [
                                s for s in saved_sessions
                                if all(
                                    w in s.get("title", "").lower()
                                    or w in s.get("content_preview", "").lower()
                                    for w in search_words
                                )
                            ]
                        else:
                            filtered_sessions = saved_sessions

                        history_box = st.container(height=320)
                        with history_box:
                            if filtered_sessions:
                                for s in filtered_sessions:
                                    raw_date = s.get("updated_at") or s.get("created_at") or ""
                                    short_date = raw_date[:10] if raw_date else ""
                                    s_col1, s_col2, s_col3 = st.columns([3, 1, 1])
                                    with s_col1:
                                        title_text = s["title"][:35]
                                        if short_date:
                                            st.caption(f"**{title_text}** &nbsp; `{short_date}`")
                                        else:
                                            st.caption(f"**{title_text}**")
                                    with s_col2:
                                        if st.button("Load", key=f"load_{s['id']}", use_container_width=True):
                                            try:
                                                detail_resp = session_api.get(f"{CHAT_API}/sessions/{s['id']}", timeout=60)
                                                if detail_resp.status_code == 200:
                                                    detail = detail_resp.json()
                                                    st.session_state.chat_session_id = s["id"]
                                                    st.session_state.chat_messages = [
                                                        {"role": m["role"], "content": m["content"],
                                                         "suggestions": m.get("metadata", {}).get("suggestions", []) if isinstance(m.get("metadata"), dict) else [],
                                                         "sources": m.get("metadata", {}).get("sources", []) if isinstance(m.get("metadata"), dict) else []}
                                                        for m in detail.get("messages", [])
                                                    ]
                                                    st.rerun()
                                            except Exception:
                                                st.error("Failed to load session")
                                    with s_col3:
                                        if st.button("Del", key=f"del_{s['id']}", use_container_width=True):
                                            try:
                                                session_api.delete(f"{CHAT_API}/sessions/{s['id']}", timeout=60)
                                                st.rerun()
                                            except Exception:
                                                pass
                            else:
                                st.caption("No conversations match your search.")
                    else:
                        st.caption("No saved conversations yet.")
            except requests.exceptions.ConnectionError:
                st.caption("API offline - session history unavailable.")
            except Exception:
                st.caption("Could not load session history.")

# Tab 2: Search Knowledge Base
with tab2:
    search_mode = st.radio(
        "Search mode:",
        options=["📄 Browse Documents", "🔍 Search Vector DB"],
        horizontal=True,
        key="kb_search_mode",
        label_visibility="collapsed"
    )

    if search_mode == "📄 Browse Documents":
        # --- Browse full documents from KB (semantic search) ---
        search_col, btn_col = st.columns([5, 1])
        with search_col:
            filter_keyword = st.text_input(
                "Search documents by meaning or keyword:",
                placeholder="e.g., machine learning, pandas, neural networks",
                key="doc_filter_keyword"
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔍 Search", key="doc_search_btn", use_container_width=True)

        # Fetch document list — semantic search when keyword provided, list all otherwise
        if filter_keyword and filter_keyword.strip():
            doc_list = fetch_kb_documents_semantic(filter_keyword.strip())
        else:
            doc_list = fetch_kb_documents()

        if doc_list:
            # Build display labels: "Note 1: Title (topic)"
            doc_labels = [
                f"Note {i+1}: {doc['title']}  —  {doc.get('topic', 'general')}"
                for i, doc in enumerate(doc_list)
            ]

            st.caption(f"{len(doc_list)} document{'s' if len(doc_list) != 1 else ''} found")

            selected_label = st.selectbox(
                "Select a document to view:",
                options=doc_labels,
                index=None,
                placeholder="Choose a document...",
                key="doc_selector",
                label_visibility="collapsed"
            )

            if selected_label:
                # Get the selected document's index and ID
                selected_idx = doc_labels.index(selected_label)
                selected_doc_meta = doc_list[selected_idx]
                selected_doc_id = selected_doc_meta['id']

                # Fetch full document content
                doc_data = fetch_document_content(selected_doc_id)

                if doc_data:
                    # Show document info bar
                    info_cols = st.columns([2, 2, 1])
                    with info_cols[0]:
                        st.caption(f"📂 Topic: {doc_data.get('topic', 'general')}")
                    with info_cols[1]:
                        date_str = doc_data.get('date_added', '')
                        if date_str:
                            try:
                                dt = datetime.fromisoformat(date_str)
                                date_str = dt.strftime("%b %d, %Y")
                            except:
                                pass
                        st.caption(f"📅 Added: {date_str}")
                    with info_cols[2]:
                        st.caption(f"📝 {doc_data.get('word_count', 0)} words")

                    # Display full document content in scrollable styled container
                    # Use white-space:pre-wrap to preserve original formatting (line breaks, indentation)
                    escaped_content = html_module.escape(doc_data["content"])
                    st.markdown(
                        f'<div class="notes-container" style="max-height:500px; overflow-y:auto; white-space:pre-wrap;">{escaped_content}</div>',
                        unsafe_allow_html=True
                    )

                    # Action buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        st.download_button(
                            "⬇️ Download",
                            data=doc_data['content'],
                            file_name=f"{doc_data.get('title', 'document')}.txt",
                            mime="text/plain",
                            key=f"download_doc_{selected_doc_id}",
                            use_container_width=True
                        )
                    with btn_col2:
                        if st.button("📋 Copy Text", key=f"copy_doc_{selected_doc_id}", use_container_width=True):
                            st.text_area(
                                "Select all and copy (Ctrl+A, Ctrl+C):",
                                value=doc_data['content'],
                                height=150,
                                key=f"copy_area_doc_{selected_doc_id}",
                                label_visibility="collapsed"
                            )
                    with btn_col3:
                        raw_url = f"{API_BASE_URL}/documents/{selected_doc_id}/raw"
                        st.markdown(
                            f'<a href="{raw_url}" target="_blank" style="display:inline-block; width:100%; text-align:center; '
                            f'padding:0.4rem 0; border:1px solid rgba(128,128,128,0.3); border-radius:0.5rem; '
                            f'text-decoration:none; color:var(--text-color); font-size:0.875rem;">'
                            f'📄 View as Text</a>',
                            unsafe_allow_html=True
                        )
                else:
                    st.error("Could not load document content.")
        else:
            if filter_keyword:
                st.info(f"No documents found matching '{filter_keyword}'")

    else:
        # --- Search chunks in Vector DB ---
        search_query = st.text_input("Enter search query:", placeholder="e.g., neural networks", key="chunk_search_query")

        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results:", 1, 10, 5, key="chunk_num_results")
        with col2:
            threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05, key="chunk_threshold")

        if st.button("🔍 Search", key="search_kb_chunks"):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        response = get_api_session().post(
                            f"{API_BASE_URL}/search-kb",
                            json={
                                "query": search_query,
                                "k": num_results,
                                "threshold": threshold
                            }
                        )

                        if response.status_code == 200:
                            result = response.json()

                            if result['success']:
                                st.success(f"Found {result['count']} matching chunks")

                                for idx, doc in enumerate(result['results'], 1):
                                    meta = doc.get('metadata', {})
                                    title = meta.get('title', 'Untitled')
                                    topic = meta.get('topic', '')
                                    source = meta.get('source', '')
                                    score = doc.get('score', 0)
                                    chunk_idx = meta.get('chunk_index', '')
                                    total_chunks = meta.get('total_chunks', '')

                                    # Build header with key info
                                    header = f"Note {idx}: {title}"
                                    if topic:
                                        header += f"  —  {topic}"
                                    header += f"  (Score: {score:.2f})"

                                    with st.expander(header, expanded=(idx == 1)):
                                        # Display chunk text cleanly
                                        st.markdown(
                                            f'<div class="notes-container">{doc["content"]}</div>',
                                            unsafe_allow_html=True
                                        )
                                        # Show only essential metadata inline
                                        meta_parts = []
                                        if source:
                                            meta_parts.append(f"Source: {source}")
                                        if chunk_idx != '' and total_chunks != '':
                                            meta_parts.append(f"Chunk {chunk_idx + 1} of {total_chunks}")
                                        if meta_parts:
                                            st.caption(" · ".join(meta_parts))
                            else:
                                st.error(f"Error: {result.get('error')}")
                        else:
                            st.error(f"API Error: {response.status_code}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a search query")

# Tab 3: Update Knowledge Base
with tab3:
    st.markdown("""
    Add new documents to the knowledge base. Each document should have:
    - **content**: The main text content
    - **source**: Where it came from
    - **title**: Document title
    - **topic**: Category/topic
    """)
    
    # Manual document input
    with st.form("add_document"):
        doc_title = st.text_input("Document Title")
        doc_topic = st.text_input("Topic", value="general")
        doc_source = st.text_input("Source", value="manual")
        doc_content = st.text_area("Content", height=200)
        
        submitted = st.form_submit_button("📤 Add to Knowledge Base")
        
        if submitted:
            if doc_content and doc_title:
                try:
                    document = {
                        "content": doc_content,
                        "title": doc_title,
                        "topic": doc_topic,
                        "source": doc_source
                    }
                    
                    response = get_api_session().post(
                        f"{API_BASE_URL}/update-kb",
                        json={"documents": [document]}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result['success']:
                            st.success(f"✅ Added {result['documents_added']} document(s)")
                        else:
                            st.error(f"Error: {result.get('error')}")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please provide both title and content")

# Tab 4: Study Mode
with tab4:
    # Sub-tabs for different study features
    study_tab1, study_tab2, study_tab3 = st.tabs(["🃏 Flashcards", "📋 Quizzes", "📊 Progress"])

    # Flashcards Sub-tab
    with study_tab1:
        st.subheader("Flashcard Study")

        # Initialize session state for flashcards
        if 'current_flashcard_set' not in st.session_state:
            st.session_state.current_flashcard_set = None
        if 'current_card_index' not in st.session_state:
            st.session_state.current_card_index = 0
        if 'show_answer' not in st.session_state:
            st.session_state.show_answer = False

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Generate Flashcards")
            fc_topic = st.text_input("Topic:", key="fc_topic", placeholder="e.g., Machine Learning")
            fc_num = st.slider("Number of cards:", 5, 20, 10, key="fc_num")
            fc_content = st.text_area("Content to study:", height=150, key="fc_content",
                                       placeholder="Paste your notes or study content here...")

            if st.button("🎯 Generate Flashcards", type="primary"):
                if fc_content and fc_topic:
                    with st.spinner("Generating flashcards..."):
                        try:
                            response = get_api_session().post(
                                f"{API_BASE_URL}/study/flashcards/generate",
                                json={
                                    "content": fc_content,
                                    "topic": fc_topic,
                                    "num_cards": fc_num
                                },
                                timeout=60
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result['success']:
                                    st.session_state.current_flashcard_set = result['flashcard_set']
                                    st.session_state.current_card_index = 0
                                    st.session_state.show_answer = False
                                    st.success(f"Generated {len(result['flashcard_set']['cards'])} flashcards!")
                                else:
                                    st.error(result.get('error', 'Failed to generate flashcards'))
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please provide both topic and content")

            st.divider()

            # Load existing flashcard sets - Using cached function (30 sec TTL)
            st.markdown("#### Or Load Existing Set")
            flashcard_sets = fetch_flashcard_sets() if check_api_health() else []
            if flashcard_sets:
                set_options = {f"{s['name']} ({s['card_count']} cards)": s['id'] for s in flashcard_sets}
                selected = st.selectbox("Select a set:", [""] + list(set_options.keys()))
                if selected and st.button("Load Set"):
                    set_id = set_options[selected]
                    try:
                        session = get_api_session()
                        resp = session.get(f"{API_BASE_URL}/study/flashcards/sets/{set_id}")
                        if resp.status_code == 200:
                            st.session_state.current_flashcard_set = resp.json()['flashcard_set']
                            st.session_state.current_card_index = 0
                            st.session_state.show_answer = False
                            st.rerun()
                    except:
                        st.error("Failed to load flashcard set")
            else:
                st.info("No flashcard sets yet. Generate some!")

            st.divider()

            # Export to Anki - Reuses cached flashcard_sets (no duplicate API call)
            st.markdown("#### 📤 Export to Anki")
            # Reuse the same cached data from fetch_flashcard_sets()
            export_sets = fetch_flashcard_sets()
            if export_sets:
                # Export specific set
                st.markdown("**Export a specific set:**")
                export_options = {f"{s['name']} ({s['card_count']} cards)": s['id'] for s in export_sets}
                selected_export = st.selectbox("Choose set to export:", [""] + list(export_options.keys()), key="export_select")

                if selected_export:
                    set_id = export_options[selected_export]
                    export_url = f"{API_BASE_URL}/study/flashcards/export/{set_id}/anki"
                    st.markdown(f"[⬇️ Download {selected_export.split(' (')[0]} for Anki]({export_url})", unsafe_allow_html=False)

                # Export all sets
                st.markdown("**Or export all sets:**")
                all_export_url = f"{API_BASE_URL}/study/flashcards/export/all/anki"
                total_cards = sum(s['card_count'] for s in export_sets)
                st.markdown(f"[⬇️ Download All Flashcards ({total_cards} cards)]({all_export_url})", unsafe_allow_html=False)
                st.caption("💡 Import the .txt file into Anki to study anywhere!")
            else:
                st.info("No flashcards to export yet")

        with col2:
            st.markdown("#### Study Cards")

            if st.session_state.current_flashcard_set:
                cards = st.session_state.current_flashcard_set['cards']
                total_cards = len(cards)

                if total_cards > 0:
                    idx = st.session_state.current_card_index
                    card = cards[idx]

                    # Progress bar
                    st.progress((idx + 1) / total_cards)
                    st.caption(f"Card {idx + 1} of {total_cards}")

                    # Flashcard display
                    with st.container():
                        st.markdown("""
                        <style>
                        .flashcard {
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 2rem;
                            border-radius: 15px;
                            min-height: 200px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            text-align: center;
                            font-size: 1.2rem;
                            margin: 1rem 0;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        if not st.session_state.show_answer:
                            st.markdown(f"<div class='flashcard'><strong>Q:</strong> {card['front']}</div>", unsafe_allow_html=True)
                            if st.button("🔄 Show Answer", use_container_width=True):
                                st.session_state.show_answer = True
                                st.rerun()
                        else:
                            st.markdown(f"<div class='flashcard' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'><strong>A:</strong> {card['back']}</div>", unsafe_allow_html=True)

                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅ Got it!", use_container_width=True, type="primary"):
                                    # Record correct review
                                    try:
                                        get_api_session().post(
                                            f"{API_BASE_URL}/study/flashcards/review",
                                            json={
                                                "set_id": st.session_state.current_flashcard_set['id'],
                                                "card_id": card['id'],
                                                "correct": True
                                            }
                                        )
                                    except:
                                        pass
                                    st.session_state.show_answer = False
                                    if idx < total_cards - 1:
                                        st.session_state.current_card_index += 1
                                    st.rerun()
                            with col_b:
                                if st.button("🔄 Review Again", use_container_width=True):
                                    try:
                                        get_api_session().post(
                                            f"{API_BASE_URL}/study/flashcards/review",
                                            json={
                                                "set_id": st.session_state.current_flashcard_set['id'],
                                                "card_id": card['id'],
                                                "correct": False
                                            }
                                        )
                                    except:
                                        pass
                                    st.session_state.show_answer = False
                                    if idx < total_cards - 1:
                                        st.session_state.current_card_index += 1
                                    st.rerun()

                    # Navigation
                    st.divider()
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                    with nav_col1:
                        if st.button("⬅️ Previous") and idx > 0:
                            st.session_state.current_card_index -= 1
                            st.session_state.show_answer = False
                            st.rerun()
                    with nav_col2:
                        if st.button("🔀 Shuffle Cards"):
                            random.shuffle(cards)
                            st.session_state.current_flashcard_set['cards'] = cards
                            st.session_state.current_card_index = 0
                            st.session_state.show_answer = False
                            st.rerun()
                    with nav_col3:
                        if st.button("➡️ Next") and idx < total_cards - 1:
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            st.rerun()
            else:
                st.info("Generate flashcards or load an existing set to start studying!")

    # Quizzes Sub-tab
    with study_tab2:
        st.subheader("Quiz Mode")

        # Initialize quiz session state
        if 'current_quiz' not in st.session_state:
            st.session_state.current_quiz = None
        if 'quiz_attempt_id' not in st.session_state:
            st.session_state.quiz_attempt_id = None
        if 'quiz_answers' not in st.session_state:
            st.session_state.quiz_answers = {}
        if 'quiz_submitted' not in st.session_state:
            st.session_state.quiz_submitted = False

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Generate Quiz")
            quiz_topic = st.text_input("Topic:", key="quiz_topic", placeholder="e.g., Neural Networks")
            quiz_num = st.slider("Number of questions:", 3, 15, 5, key="quiz_num")
            quiz_content = st.text_area("Content:", height=150, key="quiz_content",
                                         placeholder="Paste content to generate quiz from...")

            if st.button("🎲 Generate Quiz", type="primary"):
                if quiz_content and quiz_topic:
                    with st.spinner("Generating quiz..."):
                        try:
                            response = get_api_session().post(
                                f"{API_BASE_URL}/study/quizzes/generate",
                                json={
                                    "content": quiz_content,
                                    "topic": quiz_topic,
                                    "num_questions": quiz_num
                                },
                                timeout=60
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result['success']:
                                    # Start attempt
                                    start_resp = get_api_session().post(
                                        f"{API_BASE_URL}/study/quizzes/{result['quiz']['id']}/start"
                                    )
                                    if start_resp.status_code == 200:
                                        start_result = start_resp.json()
                                        st.session_state.current_quiz = result['quiz']
                                        st.session_state.quiz_attempt_id = start_result['attempt_id']
                                        st.session_state.quiz_answers = {}
                                        st.session_state.quiz_submitted = False
                                        st.success(f"Generated quiz with {len(result['quiz']['questions'])} questions!")
                                        st.rerun()
                                else:
                                    st.error(result.get('error', 'Failed to generate quiz'))
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please provide both topic and content")

            st.divider()

            # Load existing quizzes - Using cached function (30 sec TTL)
            st.markdown("#### Or Load Existing Quiz")
            quizzes_list = fetch_quizzes_list() if check_api_health() else []
            if quizzes_list:
                quiz_options = {f"{q['title']} ({q['question_count']} Q)": q['id'] for q in quizzes_list}
                selected = st.selectbox("Select a quiz:", [""] + list(quiz_options.keys()), key="load_quiz")
                if selected and st.button("Load Quiz"):
                    quiz_id = quiz_options[selected]
                    try:
                        session = get_api_session()
                        resp = session.get(f"{API_BASE_URL}/study/quizzes/{quiz_id}")
                        if resp.status_code == 200:
                            quiz_data = resp.json()['quiz']
                            start_resp = session.post(f"{API_BASE_URL}/study/quizzes/{quiz_id}/start")
                            if start_resp.status_code == 200:
                                st.session_state.current_quiz = quiz_data
                                st.session_state.quiz_attempt_id = start_resp.json()['attempt_id']
                                st.session_state.quiz_answers = {}
                                st.session_state.quiz_submitted = False
                                st.rerun()
                    except:
                        st.error("Failed to load quiz")
            else:
                st.info("No quizzes yet. Generate one!")

        with col2:
            st.markdown("#### Quiz Questions")

            if st.session_state.current_quiz and not st.session_state.quiz_submitted:
                quiz = st.session_state.current_quiz
                questions = quiz['questions']

                st.markdown(f"**{quiz['title']}** - {len(questions)} questions")
                st.divider()

                for i, q in enumerate(questions):
                    st.markdown(f"**Q{i+1}.** {q['question']}")

                    answer_key = f"q_{q['id']}"
                    options = q['options']

                    selected = st.radio(
                        "Select your answer:",
                        options,
                        key=answer_key,
                        index=None
                    )

                    if selected:
                        st.session_state.quiz_answers[q['id']] = selected

                    st.divider()

                if st.button("📝 Submit Quiz", type="primary", use_container_width=True):
                    if len(st.session_state.quiz_answers) < len(questions):
                        st.warning("Please answer all questions before submitting")
                    else:
                        with st.spinner("Submitting quiz..."):
                            # Submit all answers
                            for q_id, answer in st.session_state.quiz_answers.items():
                                get_api_session().post(
                                    f"{API_BASE_URL}/study/quizzes/submit-answer",
                                    json={
                                        "quiz_id": quiz['id'],
                                        "attempt_id": st.session_state.quiz_attempt_id,
                                        "question_id": q_id,
                                        "answer": answer
                                    }
                                )

                            # Complete the quiz
                            complete_resp = get_api_session().post(
                                f"{API_BASE_URL}/study/quizzes/complete",
                                json={
                                    "quiz_id": quiz['id'],
                                    "attempt_id": st.session_state.quiz_attempt_id
                                }
                            )

                            if complete_resp.status_code == 200:
                                result = complete_resp.json()
                                st.session_state.quiz_submitted = True
                                st.session_state.quiz_results = result
                                st.rerun()

            elif st.session_state.quiz_submitted:
                results = st.session_state.quiz_results

                # Display score
                score = results.get('score', 0)
                correct_count = results.get('correct_count', 0)
                total_questions = results.get('total_questions', 0)

                st.markdown("### 📊 Quiz Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if score >= 80:
                        st.success(f"🎉 Excellent!\n\n**{score:.1f}%**")
                    elif score >= 60:
                        st.info(f"👍 Good job!\n\n**{score:.1f}%**")
                    else:
                        st.warning(f"📚 Keep studying!\n\n**{score:.1f}%**")
                with col2:
                    st.metric("Correct Answers", f"{correct_count} / {total_questions}")
                with col3:
                    st.metric("Incorrect", f"{total_questions - correct_count}")

                st.markdown("---")

                # Display detailed results for each question
                st.markdown("### 📝 Detailed Review")

                detailed_results = results.get('detailed_results', [])
                if detailed_results:
                    for idx, question_result in enumerate(detailed_results, 1):
                        is_correct = question_result['is_correct']

                        # Question header with status icon
                        if is_correct:
                            st.markdown(f"#### Question {idx} ✅")
                        else:
                            st.markdown(f"#### Question {idx} ❌")

                        # Question text
                        st.markdown(f"**{question_result['question_text']}**")

                        # Display options with highlighting
                        options = question_result['options']
                        user_answer = question_result['user_answer']
                        correct_answer = question_result['correct_answer']

                        for opt in options:
                            opt_clean = opt.strip()

                            # Highlight correct answer in green
                            if opt_clean == correct_answer:
                                if is_correct:
                                    st.success(f"✓ **{opt_clean}** (Correct Answer - Your Choice)")
                                else:
                                    st.success(f"✓ **{opt_clean}** (Correct Answer)")

                            # Highlight incorrect user answer in red
                            elif opt_clean == user_answer and not is_correct:
                                st.error(f"✗ **{opt_clean}** (Your Choice)")

                            # Regular option
                            else:
                                st.write(f"  {opt_clean}")

                        # Show explanation
                        explanation = question_result.get('explanation')
                        if explanation:
                            st.info(f"**Explanation:** {explanation}")

                        st.markdown("---")

                st.markdown("")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Try Again", use_container_width=True):
                        st.session_state.current_quiz = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.rerun()
                with col2:
                    if st.button("📚 New Quiz", use_container_width=True):
                        st.session_state.current_quiz = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_attempt_id = None
                        st.rerun()
            else:
                st.info("Generate a quiz or load an existing one to start!")

    # Progress Sub-tab - Using cached function (60 sec TTL)
    with study_tab3:
        st.subheader("Study Progress Dashboard")

        progress = fetch_detailed_progress() if check_api_health() else None

        if progress:
            # Overall Stats
            st.markdown("#### Overall Statistics")
            stats = progress.get('overall_stats', {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Notes Generated", stats.get('total_notes_generated', 0))
            with col2:
                st.metric("Flashcards Reviewed", stats.get('total_flashcards_reviewed', 0))
            with col3:
                st.metric("Quizzes Completed", stats.get('total_quizzes_completed', 0))
            with col4:
                st.metric("Topics Studied", stats.get('topics_studied', 0))

            st.divider()

            # Streak Information
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔥 Study Streak")
                streak = progress.get('streak', {})
                st.metric("Current Streak", f"{streak.get('current_streak', 0)} days")
                st.metric("Longest Streak", f"{streak.get('longest_streak', 0)} days")

            with col2:
                st.markdown("#### 📈 Accuracy")
                st.metric("Flashcard Accuracy", f"{stats.get('flashcard_accuracy', 0):.1f}%")
                st.metric("Quiz Accuracy", f"{stats.get('quiz_accuracy', 0):.1f}%")

            st.divider()

            # Topic Rankings
            st.markdown("#### 🏆 Topic Mastery")
            rankings = progress.get('topic_rankings', [])
            if rankings:
                for rank in rankings[:5]:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{rank['topic']}**")
                    with col2:
                        st.write(f"Level: {rank['mastery_level']}")
                    with col3:
                        st.write(f"Notes: {rank['notes_generated']}")
            else:
                st.info("No topics studied yet. Start learning!")

            st.divider()

            # Weekly Summary
            st.markdown("#### 📅 This Week")
            weekly = progress.get('weekly_summary', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Notes This Week", weekly.get('notes_generated', 0))
            with col2:
                st.metric("Flashcards Reviewed", weekly.get('flashcards_reviewed', 0))
            with col3:
                st.metric("Active Days", f"{weekly.get('active_days', 0)}/7")

            st.divider()

            # Recent Activities
            st.markdown("#### 📝 Recent Activity")
            activities = progress.get('recent_activities', [])
            if activities:
                for act in activities[:5]:
                    st.text(f"• {act.get('summary', 'Activity')} - {act.get('timestamp', '')[:10]}")
            else:
                st.info("No recent activity")
        else:
            st.error("Could not load progress data")
            st.info("Make sure the API server is running")

# Footer
st.markdown("""
<hr style="margin-top: 5rem; margin-bottom: 0.5rem; border: none; border-top: 1px solid rgba(128,128,128,0.2);" />
<div style='text-align: center; color: #666; font-size: 0.7rem; padding-bottom: 2rem;'>
    <p>EduNotes v2.0 | Multi-Agent Study Assistant</p>
</div>
""", unsafe_allow_html=True)