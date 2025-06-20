"""
Streamlit web application for the Document Q&A System.
Provides an intuitive interface for document upload and question-answering.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Import our custom modules
from src.qa_chain import DocumentQAChain
from src.utils import (
    setup_environment, 
    format_response_for_display, 
    get_file_info,
    format_file_size
)

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e9ecef;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}


def setup_sidebar():
    """Setup the sidebar with controls and information."""
    st.sidebar.title("🔧 System Controls")
    
    # Environment setup
    with st.sidebar.expander("⚙️ Environment Setup", expanded=False):
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set!")
        
        if st.button("🔄 Test Environment"):
            try:
                setup_environment()
                st.success("Environment is properly configured!")
            except Exception as e:
                st.error(f"Environment error: {str(e)}")
    
    # Model configuration
    with st.sidebar.expander("🤖 Model Settings", expanded=False):
        model_name = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in responses"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum length of responses"
        )
        
        max_docs_retrieve = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant documents to use for context"
        )
        
        if st.button("🔄 Update Model Settings"):
            st.session_state.qa_chain = DocumentQAChain(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_docs_retrieve=max_docs_retrieve
            )
            st.success("Model settings updated!")
    
    # System statistics
    if st.session_state.documents_loaded:
        with st.sidebar.expander("📊 System Statistics", expanded=False):
            if st.session_state.qa_chain:
                stats = st.session_state.qa_chain.get_system_stats()
                st.json(stats)
    
    # Conversation controls
    with st.sidebar.expander("💬 Conversation", expanded=False):
        if st.button("🗑️ Clear Conversation"):
            if st.session_state.qa_chain:
                st.session_state.qa_chain.clear_conversation_history()
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
        
        if st.button("📜 Show History"):
            if st.session_state.qa_chain:
                history = st.session_state.qa_chain.get_conversation_history()
                for i, exchange in enumerate(history, 1):
                    st.write(f"**Q{i}:** {exchange['question']}")
                    st.write(f"**A{i}:** {exchange['answer']}")
                    st.write("---")


def document_upload_section():
    """Handle document upload and processing."""
    st.markdown('<h2 class="sub-header">📁 Document Upload & Processing</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        # Directory input option
        documents_path = st.text_input(
            "Or specify documents directory path:",
            value="./documents",
            help="Path to directory containing documents"
        )
    
    with col2:
        st.markdown("""
        **Supported Formats:**
        - 📄 PDF files
        - 📝 Word documents (.docx)
        - 📃 Text files (.txt)
        
        **Tips:**
        - Use clear, well-formatted documents
        - Avoid scanned images in PDFs
        - Optimal file size: < 50MB each
        """)
    
    # Process uploaded files
    if uploaded_files:
        # Create temporary directory for uploaded files
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))
        
        # Display file information
        st.markdown("**Uploaded Files:**")
        for file_path in file_paths:
            file_info = get_file_info(file_path)
            st.write(f"📄 {file_info['name']} ({file_info['size_formatted']})")
        
        documents_path = str(temp_dir)
    
    # Process documents button
    if st.button("🚀 Process Documents", type="primary"):
        if not documents_path:
            st.error("Please provide a documents path or upload files")
            return
        
        try:
            # Initialize QA chain if needed
            if not st.session_state.qa_chain:
                st.session_state.qa_chain = DocumentQAChain()
            
            with st.spinner("Processing documents... This may take a few minutes."):
                # Load and process documents
                result = st.session_state.qa_chain.load_documents(documents_path)
                
                if result["status"] == "success":
                    st.session_state.documents_loaded = True
                    
                    # Display success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Documents Processed Successfully!</h4>
                        <p><strong>Documents loaded:</strong> {result['documents_loaded']}</p>
                        <p><strong>Total chunks:</strong> {result['document_stats']['total_documents']}</p>
                        <p><strong>File types:</strong> {', '.join(result['document_stats']['file_types'].keys())}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detailed statistics
                    with st.expander("📊 Detailed Statistics"):
                        st.json(result)
                        
                else:
                    st.error(f"Error processing documents: {result['error']}")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")


def question_answering_section():
    """Handle question-answering interface."""
    if not st.session_state.documents_loaded:
        st.markdown("""
        <div class="info-box">
            <h4>📚 Ready to Answer Questions!</h4>
            <p>Please upload and process documents first to start asking questions.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<h2 class="sub-header">❓ Ask Questions</h2>', 
                unsafe_allow_html=True)
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is the main topic of the documents? Who are the key stakeholders? What are the financial projections?",
        help="Ask specific questions for better results"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_button = st.button("🤔 Ask Question", type="primary")
    
    with col2:
        include_sources = st.checkbox("Include Sources", value=True)
    
    # Process question
    if ask_button and question:
        try:
            with st.spinner("Thinking... 🧠"):
                response = st.session_state.qa_chain.ask(
                    question, 
                    include_sources=include_sources
                )
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(response['answer'])
                
                # Display sources if available
                if include_sources and response.get('sources'):
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(response['sources'], 1):
                        with st.expander(f"📄 Source {i}: {source['filename']}"):
                            st.markdown(f"**File Type:** {source['file_type']}")
                            st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                            st.markdown(f"**Content Preview:**")
                            st.markdown(f"```\n{source['content_preview']}\n```")
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'timestamp': response['timestamp']
                })
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What are the main topics covered in the documents?",
            "Who are the key people or organizations mentioned?",
            "What are the important dates or deadlines?",
            "Can you summarize the key findings?",
            "What recommendations are made?",
            "What are the potential risks or challenges mentioned?"
        ]
        
        for i, sample in enumerate(sample_questions):
            if st.button(f"📝 {sample}", key=f"sample_{i}"):
                st.rerun()


def conversation_history_section():
    """Display conversation history."""
    if st.session_state.conversation_history:
        st.markdown('<h2 class="sub-header">💬 Conversation History</h2>', 
                    unsafe_allow_html=True)
        
        for i, exchange in enumerate(reversed(st.session_state.conversation_history), 1):
            with st.expander(f"Exchange {len(st.session_state.conversation_history) - i + 1}: {exchange['question'][:50]}..."):
                st.markdown(f"**Question:** {exchange['question']}")
                st.markdown(f"**Answer:** {exchange['answer']}")
                st.markdown(f"**Time:** {exchange['timestamp']}")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>🚀 <strong>Intelligent Document Analysis</strong> - Upload your documents and ask questions to get AI-powered answers with source citations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Document Processing", "❓ Q&A Interface", "💬 Conversation"])
    
    with tab1:
        document_upload_section()
    
    with tab2:
        question_answering_section()
    
    with tab3:
        conversation_history_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, LangChain, and OpenAI | 
        <a href='https://github.com/yourusername/document-qa-system'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
