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
tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate Notes", "🔍 Search Knowledge Base", "📤 Update Knowledge Base", "📖 Study Mode"])

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
                if score >= 80:
                    st.success(f"🎉 Excellent! Score: {score:.1f}%")
                elif score >= 60:
                    st.info(f"👍 Good job! Score: {score:.1f}%")
                else:
                    st.warning(f"📚 Keep studying! Score: {score:.1f}%")

                st.metric("Correct Answers", f"{results.get('correct_count', 0)} / {results.get('total_questions', 0)}")

                if st.button("🔄 Try Again"):
                    st.session_state.current_quiz = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
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