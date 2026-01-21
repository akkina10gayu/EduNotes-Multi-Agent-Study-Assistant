"""
Streamlit UI for EduNotes
Version 2.0 - With Study Features (Flashcards, Quizzes, Progress)
"""
import streamlit as st
import requests
import json
from datetime import datetime
import time
import random

# Page configuration
st.set_page_config(
    page_title="EduNotes - Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"

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

# Font size mapping
font_sizes = {
    'small': {'base': '14px', 'header': '2.5rem', 'notes': '0.9rem'},
    'medium': {'base': '16px', 'header': '3rem', 'notes': '1rem'},
    'large': {'base': '18px', 'header': '3.5rem', 'notes': '1.1rem'}
}

current_size = font_sizes[st.session_state.font_size]

# Apply unified CSS with dynamic font sizing
st.markdown(f"""
<style>
    /* Base styling */
    html, body, .stApp {{
        font-size: {current_size['base']};
    }}

    .main-header {{
        font-size: {current_size['header']};
        color: #6CA0DC;
        text-align: center;
        margin-bottom: 2rem;
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
    }}

    /* Improve readability */
    .stMarkdown p, .stText {{
        line-height: 1.6;
    }}

    /* Button styling */
    .stButton>button {{
        font-size: {current_size['base']};
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 EduNotes Study Assistant</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # API Status Check
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")

            # Check if API key is configured
            import os
            env_path = os.path.join(os.getcwd(), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    env_content = f.read()
                    if 'GROQ_API_KEY=your_groq_key_here' in env_content or 'GROQ_API_KEY=' not in env_content:
                        st.warning("⚠️ API key not configured")
                        st.caption("See Help below for setup")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.caption("See Help below to start API")

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

    # System Stats
    if st.button("📊 View System Stats"):
        try:
            response = requests.get(f"{API_BASE_URL}/stats", timeout=3)
            if response.status_code == 200:
                stats = response.json()

                # Display Knowledge Base Stats
                st.markdown("**📚 Knowledge Base**")
                kb_stats = stats.get('knowledge_base', {})
                st.markdown(f"- **Total Documents:** {kb_stats.get('total_documents', 0):,}")
                st.markdown(f"- **Collection:** {kb_stats.get('collection_name', 'N/A')}")
                st.markdown(f"- **Storage:** {kb_stats.get('persist_directory', 'N/A')}")

                st.markdown("")

                # Display Agents Status
                st.markdown("**🤖 AI Agents**")
                agents = stats.get('agents', {})
                for agent_name, status in agents.items():
                    status_icon = "🟢" if status == "active" else "🔴"
                    agent_display = agent_name.replace('_', ' ').title()
                    st.markdown(f"- **{agent_display}:** {status_icon} {status.title()}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ **API Server Not Running**")
            st.warning("""
            The backend API server is not running. Please start it:

            ```bash
            uvicorn src.api.app:app --reload
            ```

            The API should run on http://localhost:8000
            """)
        except requests.exceptions.Timeout:
            st.error("⏱️ **Request Timeout** - API server is slow or unresponsive")
        except Exception as e:
            st.error(f"❌ Error fetching stats: {str(e)[:100]}")
    
    st.divider()

    # Note History
    st.header("📜 Note History")
    if st.session_state.note_history:
        st.markdown(f"*Last {len(st.session_state.note_history)} notes*")
        for idx, note_item in enumerate(reversed(st.session_state.note_history)):
            with st.expander(f"{note_item['timestamp']} - {note_item['query'][:30]}..."):
                st.markdown(f"**Query:** {note_item['query'][:100]}...")
                st.markdown(f"**Type:** {note_item['type']}")
                if st.button(f"📋 View Notes", key=f"history_{idx}"):
                    st.session_state.selected_history = note_item
                    st.rerun()
    else:
        st.info("📝 No notes in this session yet. Generate a note to see it appear here!")

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

# Quick Stats Dashboard
st.markdown("### 📊 Your Study Stats")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

try:
    # Get progress data
    progress_resp = requests.get(f"{API_BASE_URL}/study/progress", timeout=3)
    flashcard_resp = requests.get(f"{API_BASE_URL}/study/flashcards/sets", timeout=3)

    notes_generated = len(st.session_state.get('note_history', []))

    if progress_resp.status_code == 200:
        progress_data = progress_resp.json()
        total_flashcards = progress_data.get('flashcard_reviews', 0)
        total_quizzes = progress_data.get('quiz_attempts', 0)
        current_streak = progress_data.get('current_streak', 0)
    else:
        total_flashcards = 0
        total_quizzes = 0
        current_streak = 0

    if flashcard_resp.status_code == 200:
        flashcard_data = flashcard_resp.json()
        flashcard_sets = len(flashcard_data.get('sets', []))
    else:
        flashcard_sets = 0

    with stat_col1:
        st.metric("📝 Notes Generated", notes_generated)
    with stat_col2:
        st.metric("🃏 Flashcard Sets", flashcard_sets)
    with stat_col3:
        st.metric("📋 Quizzes Taken", total_quizzes)
    with stat_col4:
        st.metric("🔥 Study Streak", f"{current_streak} days")

except:
    with stat_col1:
        st.metric("📝 Notes Generated", len(st.session_state.get('note_history', [])))
    with stat_col2:
        st.metric("🃏 Flashcard Sets", "—")
    with stat_col3:
        st.metric("📋 Quizzes Taken", "—")
    with stat_col4:
        st.metric("🔥 Study Streak", "—")

st.divider()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate Notes", "🔍 Search Knowledge Base", "📤 Update Knowledge Base", "📖 Study Mode"])

# Tab 1: Generate Notes
with tab1:
    st.header("Generate Study Notes")

    # Show selected history note if any
    if 'selected_history' in st.session_state and st.session_state.selected_history:
        hist = st.session_state.selected_history
        st.info(f"📜 Viewing note from history: {hist['timestamp']}")

        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Type", hist['type'].upper())
        with col2:
            st.metric("Sources Used", hist['sources_used'])
        with col3:
            st.metric("From KB", "Yes" if hist['from_kb'] else "No")

        # Display notes
        st.markdown("### 📚 Historical Note")
        st.markdown('<div class="notes-container">', unsafe_allow_html=True)
        st.markdown(hist['notes'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download",
                data=hist['notes'],
                file_name=f"notes_history_{hist['timestamp'].replace(':', '-')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            if st.button("📋 Copy", use_container_width=True, key="copy_history"):
                st.session_state.show_copy_box_history = True
        with col3:
            if st.button("🔄 New Query", use_container_width=True):
                st.session_state.selected_history = None
                st.rerun()

        # Show copyable text area
        if st.session_state.get('show_copy_box_history', False):
            st.text_area(
                "Copy this text:",
                value=hist['notes'],
                height=150,
                key="copy_history_text"
            )

        st.divider()

    # Topic Suggestions
    st.markdown("#### 💡 Quick Topics from Knowledge Base")
    try:
        topics_resp = requests.get(f"{API_BASE_URL}/topics", timeout=3)
        if topics_resp.status_code == 200:
            topics_data = topics_resp.json()
            if topics_data.get('topics'):
                topics = topics_data['topics'][:12]  # Show first 12 topics

                # Display as clickable chips
                chip_cols = st.columns(4)
                for idx, topic in enumerate(topics):
                    col_idx = idx % 4
                    with chip_cols[col_idx]:
                        if st.button(f"📚 {topic}", key=f"topic_{idx}", use_container_width=True):
                            st.session_state.suggested_topic = topic
                            st.rerun()
    except:
        pass

    st.markdown("---")

    # Input section
    col1, col2 = st.columns([3, 1])

    # Pre-fill if topic was clicked
    default_query = st.session_state.get('suggested_topic', '')
    if default_query:
        st.session_state.suggested_topic = None  # Clear after using

    with col1:
        query_input = st.text_area(
            "Enter your query (topic, URL, or text):",
            height=100,
            value=default_query,
            placeholder="Examples:\n- Machine Learning\n- https://example.com/article\n- Paste your text here..."
        )
    
    with col2:
        st.markdown("#### Query Type")
        if query_input:
            if query_input.startswith(('http://', 'https://', 'www.')):
                st.info("🔗 URL Detected")
            elif len(query_input) > 500:
                st.info("📄 Text Detected")
            else:
                st.info("🎯 Topic Detected")

    # PDF Upload Section
    st.markdown("---")
    st.markdown("#### 📄 Or Upload a PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF file to generate notes",
        type=['pdf'],
        help="Upload research papers, textbooks, or any PDF document (max 10MB)"
    )

    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}** ({file_size_mb:.2f}MB)")

    st.markdown("---")

    # Summarization Mode Selection
    st.markdown("#### ⚙️ Output Format")
    summarization_mode = st.radio(
        "Choose how you want the content to be processed:",
        options=["paragraph_summary", "important_points", "key_highlights"],
        format_func=lambda x: {
            "paragraph_summary": "📖 Paragraph Summary",
            "important_points": "📋 Important Points",
            "key_highlights": "⚡ Key Highlights"
        }.get(x, x),
        help="""**Paragraph Summary**: Comprehensive overview in flowing paragraphs (3+ sentences each). Ideal for understanding the full context and relationships between concepts.

**Important Points**: Key information as numbered points. Each point is independent and self-contained with no duplicates. Perfect for quick review and study notes.

**Key Highlights**: Essential terms and concepts with brief definitions (half a line each). Great for creating glossaries and quick reference cards.""",
        horizontal=True,
        key="summarization_mode_selector"
    )

    # Display mode description
    mode_descriptions = {
        "paragraph_summary": "Creates a comprehensive overview with flowing paragraphs explaining concepts in detail.",
        "important_points": "Extracts distinct key points, each providing unique information without repetition.",
        "key_highlights": "Lists essential terms and concepts with very brief definitions for quick scanning."
    }
    st.caption(f"💡 {mode_descriptions.get(summarization_mode, '')}")

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate Notes", type="primary"):
        # Check if either query or PDF is provided
        if uploaded_file:
            # Process PDF file
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Validate file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 10:
                    st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is 10MB.")
                else:
                    progress_placeholder.progress(0.2)
                    status_placeholder.info("📄 Extracting text from PDF...")
                    time.sleep(0.3)

                    # Upload PDF to API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

                    progress_placeholder.progress(0.5)
                    mode_text_map = {
                        "paragraph_summary": "paragraph summary",
                        "important_points": "important points",
                        "key_highlights": "key highlights"
                    }
                    mode_text = mode_text_map.get(summarization_mode, "summary")
                    status_placeholder.info(f"🤖 Processing PDF content ({mode_text})...")

                    response = requests.post(
                        f"{API_BASE_URL}/process-pdf",
                        files=files,
                        params={"summarization_mode": summarization_mode},
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

                        if result['success']:
                            st.success(f"✅ Notes generated from PDF: {uploaded_file.name}")

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
                            if len(st.session_state.note_history) > 10:
                                st.session_state.note_history.pop(0)

                            # Display metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Source", "PDF Upload")
                            with col2:
                                st.metric("File Size", f"{file_size_mb:.2f} MB")
                            with col3:
                                st.metric("Filename", uploaded_file.name[:20])

                            # Display notes
                            st.markdown("### 📚 Generated Notes")
                            st.markdown('<div class="notes-container">', unsafe_allow_html=True)
                            st.markdown(result['notes'])
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download Notes",
                                    data=result['notes'],
                                    file_name=f"notes_{uploaded_file.name.replace('.pdf', '')}.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                            with col2:
                                if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_pdf"):
                                    st.session_state.show_copy_box = True

                            if st.session_state.get('show_copy_box', False):
                                st.markdown("**📋 Copy the text below:**")
                                st.text_area(
                                    "Select all (Ctrl+A) and copy (Ctrl+C):",
                                    value=result['notes'],
                                    height=200,
                                    key="copy_pdf_text"
                                )
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"❌ Error processing PDF: {str(e)}")

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
                if query_input.startswith(('http://', 'https://', 'www.')):
                    status_placeholder.info("🌐 Scraping web content...")
                elif len(query_input) > 500:
                    status_placeholder.info("📄 Processing text input...")
                else:
                    status_placeholder.info("🔍 Searching knowledge base...")

                # Make API request
                response = requests.post(
                    f"{API_BASE_URL}/generate-notes",
                    json={
                        "query": query_input,
                        "summarization_mode": summarization_mode
                    },
                    timeout=120
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

                        # Save to history (keep last 10)
                        note_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'query': query_input,
                            'type': result['query_type'],
                            'notes': result['notes'],
                            'sources_used': result['sources_used'],
                            'from_kb': result['from_kb']
                        }
                        st.session_state.note_history.append(note_entry)
                        if len(st.session_state.note_history) > 10:
                            st.session_state.note_history.pop(0)

                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Query Type", result['query_type'].upper())
                        with col2:
                            st.metric("Sources Used", result['sources_used'])
                        with col3:
                            st.metric("From KB", "Yes" if result['from_kb'] else "No")

                        # Display notes
                        st.markdown("### 📚 Generated Notes")
                        st.markdown('<div class="notes-container">', unsafe_allow_html=True)
                        st.markdown(result['notes'])
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Notes",
                                data=result['notes'],
                                file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        with col2:
                            if st.button("📋 Copy to Clipboard", use_container_width=True):
                                st.session_state.show_copy_box = True

                        # Show copyable text area when button clicked
                        if st.session_state.get('show_copy_box', False):
                            st.markdown("**📋 Copy the text below:**")
                            st.text_area(
                                "Select all (Ctrl+A) and copy (Ctrl+C):",
                                value=result['notes'],
                                height=200,
                                key="copy_text_area"
                            )
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ API Error: {response.status_code}")

            except requests.exceptions.Timeout:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query")

# Tab 2: Search Knowledge Base
with tab2:
    st.header("Search Knowledge Base")
    
    search_query = st.text_input("Enter search query:", placeholder="e.g., neural networks")
    
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of results:", 1, 10, 5)
    with col2:
        threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05)
    
    if st.button("🔍 Search", key="search_kb"):
        if search_query:
            with st.spinner("Searching..."):
                try:
                    response = requests.post(
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
                            st.success(f"Found {result['count']} results")
                            
                            for idx, doc in enumerate(result['results'], 1):
                                with st.expander(f"Result {idx} - Score: {doc['score']:.3f}"):
                                    st.markdown(f"**Content:** {doc['content'][:500]}...")
                                    st.json(doc['metadata'])
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
    st.header("Update Knowledge Base")
    
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
                    
                    response = requests.post(
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
    st.header("Study Mode")

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
                            response = requests.post(
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

            # Load existing flashcard sets
            st.markdown("#### Or Load Existing Set")
            try:
                response = requests.get(f"{API_BASE_URL}/study/flashcards/sets", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    if result['sets']:
                        set_options = {f"{s['name']} ({s['card_count']} cards)": s['id'] for s in result['sets']}
                        selected = st.selectbox("Select a set:", [""] + list(set_options.keys()))
                        if selected and st.button("Load Set"):
                            set_id = set_options[selected]
                            resp = requests.get(f"{API_BASE_URL}/study/flashcards/sets/{set_id}")
                            if resp.status_code == 200:
                                st.session_state.current_flashcard_set = resp.json()['flashcard_set']
                                st.session_state.current_card_index = 0
                                st.session_state.show_answer = False
                                st.rerun()
                    else:
                        st.info("No flashcard sets yet. Generate some!")
            except:
                pass

            st.divider()

            # Export to Anki
            st.markdown("#### 📤 Export to Anki")
            try:
                response = requests.get(f"{API_BASE_URL}/study/flashcards/sets", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    if result['sets']:
                        # Export specific set
                        st.markdown("**Export a specific set:**")
                        export_options = {f"{s['name']} ({s['card_count']} cards)": s['id'] for s in result['sets']}
                        selected_export = st.selectbox("Choose set to export:", [""] + list(export_options.keys()), key="export_select")

                        if selected_export:
                            set_id = export_options[selected_export]
                            export_url = f"{API_BASE_URL}/study/flashcards/export/{set_id}/anki"
                            st.markdown(f"[⬇️ Download {selected_export.split(' (')[0]} for Anki]({export_url})", unsafe_allow_html=False)

                        # Export all sets
                        st.markdown("**Or export all sets:**")
                        all_export_url = f"{API_BASE_URL}/study/flashcards/export/all/anki"
                        total_cards = sum(s['card_count'] for s in result['sets'])
                        st.markdown(f"[⬇️ Download All Flashcards ({total_cards} cards)]({all_export_url})", unsafe_allow_html=False)
                        st.caption("💡 Import the .txt file into Anki to study anywhere!")
                    else:
                        st.info("No flashcards to export yet")
            except:
                pass

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
                                        requests.post(
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
                                        requests.post(
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
                            response = requests.post(
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
                                    start_resp = requests.post(
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

            # Load existing quizzes
            st.markdown("#### Or Load Existing Quiz")
            try:
                response = requests.get(f"{API_BASE_URL}/study/quizzes", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    if result['quizzes']:
                        quiz_options = {f"{q['title']} ({q['question_count']} Q)": q['id'] for q in result['quizzes']}
                        selected = st.selectbox("Select a quiz:", [""] + list(quiz_options.keys()), key="load_quiz")
                        if selected and st.button("Load Quiz"):
                            quiz_id = quiz_options[selected]
                            resp = requests.get(f"{API_BASE_URL}/study/quizzes/{quiz_id}")
                            if resp.status_code == 200:
                                quiz_data = resp.json()['quiz']
                                start_resp = requests.post(f"{API_BASE_URL}/study/quizzes/{quiz_id}/start")
                                if start_resp.status_code == 200:
                                    st.session_state.current_quiz = quiz_data
                                    st.session_state.quiz_attempt_id = start_resp.json()['attempt_id']
                                    st.session_state.quiz_answers = {}
                                    st.session_state.quiz_submitted = False
                                    st.rerun()
                    else:
                        st.info("No quizzes yet. Generate one!")
            except:
                pass

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
                                requests.post(
                                    f"{API_BASE_URL}/study/quizzes/submit-answer",
                                    json={
                                        "quiz_id": quiz['id'],
                                        "attempt_id": st.session_state.quiz_attempt_id,
                                        "question_id": q_id,
                                        "answer": answer
                                    }
                                )

                            # Complete the quiz
                            complete_resp = requests.post(
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

    # Progress Sub-tab
    with study_tab3:
        st.subheader("Study Progress Dashboard")

        try:
            response = requests.get(f"{API_BASE_URL}/study/progress", timeout=5)

            if response.status_code == 200:
                progress = response.json()

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

        except Exception as e:
            st.error(f"Error loading progress: {str(e)}")
            st.info("Make sure the API server is running")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>EduNotes v2.0 | Multi-Agent Study Assistant</p>
    <p>Powered by LangChain, ChromaDB, and Transformers</p>
    <p>Study Features: Flashcards, Quizzes, Progress Tracking</p>
</div>
""", unsafe_allow_html=True)