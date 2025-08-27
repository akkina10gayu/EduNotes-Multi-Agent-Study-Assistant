"""
Streamlit UI for EduNotes
"""
import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="EduNotes - Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .success-message {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        color: #2E7D32;
    }
    .error-message {
        padding: 1rem;
        background-color: #FFEBEE;
        border-radius: 0.5rem;
        color: #C62828;
    }
    .notes-container {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
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
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.divider()
    
    # System Stats
    if st.button("📊 View System Stats"):
        try:
            response = requests.get(f"{API_BASE_URL}/stats")
            if response.status_code == 200:
                stats = response.json()
                st.json(stats)
        except Exception as e:
            st.error(f"Error fetching stats: {e}")
    
    st.divider()
    
    # Instructions
    st.header("📖 How to Use")
    st.markdown("""
    1. **Topic Query**: Enter a topic like "machine learning"
    2. **URL**: Paste a blog/article URL
    3. **Text**: Paste text directly for summarization
    
    The system will automatically detect the input type and generate structured notes.
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📝 Generate Notes", "🔍 Search Knowledge Base", "📤 Update Knowledge Base"])

# Tab 1: Generate Notes
with tab1:
    st.header("Generate Study Notes")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_input = st.text_area(
            "Enter your query (topic, URL, or text):",
            height=100,
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
    
    # Generate button
    if st.button("🚀 Generate Notes", type="primary"):
        if query_input:
            with st.spinner("🤖 Processing your request..."):
                try:
                    # Make API request
                    response = requests.post(
                        f"{API_BASE_URL}/generate-notes",
                        json={"query": query_input},
                        timeout=120  # Increased to 2 minutes for ML processing
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result['success']:
                            st.success("✅ Notes generated successfully!")
                            
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
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Notes (Markdown)",
                                data=result['notes'],
                                file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
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

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>EduNotes v1.0 | Multi-Agent Study Assistant</p>
    <p>Powered by LangChain, ChromaDB, and Transformers</p>
</div>
""", unsafe_allow_html=True)