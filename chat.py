import streamlit as st
import pandas as pd
import PyPDF2
import os
import re
import torch
import faiss
import base64
import json
from io import StringIO, BytesIO
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancement with Tailwind-like styles
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css');
    
    .main {
        background-color: white;
        padding: 40px !important;
        max-width: 100% !important;
    }
    
    .block-container {
        max-width: 100% !important;
        padding: 40px !important;
    }
    
    .chat-app-container {
    display: flex;
    min-height: 100vh;
    max-width: 1300px;
    margin: 0 auto;
    font-family: 'Inter', sans-serif;
    position: relative; /* เพิ่ม position */
    z-index: 1; /* เพิ่ม z-index */
    }

    
    .sidebar {
        width: 320px;
        background-color: white;
        border-right: 1px solid #e5e7eb;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        max-width: calc(100% - 320px);
    }
    
    .chat-header {
        background-color: white;
        padding: 16px 24px;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        width: 100%;
    }
    
    
    .date-divider {
        text-align: center;
        margin: 16px 0;
    }
    
    .date-divider span {
        background-color: #e5e7eb;
        color: #6b7280;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    
    .message-group {
        display: flex;
        margin-bottom: 16px;
    }
    
    .user-message-group {
        justify-content: flex-end;
    }
    
    .bot-message-group {
        justify-content: flex-start;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
    }
    
    .user-avatar {
        background-color: #3b82f6;
        color: white;
        margin-left: 12px;
        margin-right: 0;
    }
    
    .message-content {
        max-width: 70%;
    }
    
    .sender-name {
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .message-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.5;
        word-break: break-word;
    }
    
    .bot-message {
        background-color: #f3f4f6;
        border-top-left-radius: 4px;
    }
    
    .user-message {
        background-color: #3b82f6;
        color: white;
        border-top-right-radius: 4px;
    }
    
    .message-input-container {
        
        padding: 16px 24px;
        background-color: white;
        border-top: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        gap: 12px;
        
    }
    
    .message-input {
        flex: 1;
        padding: 12px 16px;
        border: 1px solid #e5e7eb;
        border-radius: 9999px;
        font-size: 14px;
        outline: none;
    }
    
    .message-input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }
    
    .send-button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 9999px;
        padding: 10px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .send-button:hover {
        background-color: #2563eb;
    }
    
    .sidebar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    
    .logo {
        font-size: 24px;
        font-weight: 700;
        color: #3b82f6;
        display: flex;
        align-items: center;
    }
    
    .logo i {
        margin-right: 8px;
    }
    
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
        margin: 16px 0 8px 0;
    }
    
    .upload-section, .kb-section {
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .wrapper {
    width: 100%;
    height: 100%;
    position: relative;
    }
    .badge {
        background-color: #e5e7eb;
        color: #6b7280;
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 9999px;
        margin-left: 8px;
    }
    .chat-container {
     max-height: 70vh;
     overflow-y: auto;
    }

    .notification-badge {
        background-color: #10b981;
        color: white;
        font-size: 10px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: absolute;
        top: -5px;
        right: -5px;
    }
    
    /* Override Streamlit buttons */
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Spinner styles */
    .loading-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    
    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-left-color: #3b82f6;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Customize file uploader */
    .uploadInputContainer {
        border: 2px dashed #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    
    /* Expandable sections */
    .sources-container {
        margin-top: 8px;
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 12px;
        font-size: 12px;
    }
    
    .source-item {
        padding: 4px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .source-item:last-child {
        border-bottom: none;
    }
    
    /* Override Streamlit elements */
    .stTextInput, .stTextArea {
        padding: 0 !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .chat-app-container {
            flex-direction: column;
        }
        
        .sidebar {
            width: 100%;
            max-width: 100%;
        }
        
        .main-content {
            max-width: 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
    
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'encoder' not in st.session_state:
    st.session_state.encoder = None

if 'last_response_sources' not in st.session_state:
    st.session_state.last_response_sources = []

# Model loading function
@st.cache_resource
def load_models():
    # Load mT5 model and tokenizer
    model_name = "google/mt5-base"  # You can also use google/mt5-small for faster but less accurate results
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load sentence transformer for embeddings
    encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Multilingual encoder
    
    return tokenizer, model, encoder

# Document processing functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text_content = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text_content += page.extract_text()
    return text_content

def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    text_content = ""
    for column in df.columns:
        text_content += f"{column}: "
        text_content += ", ".join(df[column].astype(str).tolist())
        text_content += "\n\n"
    return text_content

def extract_text_from_json(json_file):
    try:
        # Parse JSON content
        json_content = json.loads(json_file.getvalue().decode("utf-8"))
        
        # Function to recursively convert JSON to text
        def json_to_text(json_obj, prefix=""):
            result = ""
            if isinstance(json_obj, dict):
                for key, value in json_obj.items():
                    if isinstance(value, (dict, list)):
                        result += f"{prefix}{key}:\n"
                        result += json_to_text(value, prefix + "  ")
                    else:
                        result += f"{prefix}{key}: {value}\n"
            elif isinstance(json_obj, list):
                for i, item in enumerate(json_obj):
                    if isinstance(item, (dict, list)):
                        result += f"{prefix}Item {i+1}:\n"
                        result += json_to_text(item, prefix + "  ")
                    else:
                        result += f"{prefix}Item {i+1}: {item}\n"
            return result
        
        return json_to_text(json_content)
    except Exception as e:
        st.error(f"Error parsing JSON: {str(e)}")
        return ""

def preprocess_text(text):
    # Remove HTML tags if present
    text = re.sub('<[^<]+?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split by common Thai sentence endings
    thai_endings = ['[.!?]', ' ครับ', ' ค่ะ', ' นะคะ', ' นะครับ']
    pattern = '|'.join(thai_endings)
    
    # Split text into sentences
    sentences = []
    # First try regular punkt tokenizer
    try:
        sentences = sent_tokenize(text)
    except:
        # If that fails, use simple splitting
        raw_sentences = re.split(f'({pattern})', text)
        current_sentence = ''
        
        for part in raw_sentences:
            current_sentence += part
            if re.search(pattern, part):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ''
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
    
    # Remove empty sentences and clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def update_knowledge_base():
    # Create embeddings for the knowledge base
    if st.session_state.knowledge_base:
        with st.spinner("Updating knowledge base... This may take a moment."):
            document_embeddings = st.session_state.encoder.encode(st.session_state.knowledge_base)
            
            # Convert to numpy array if it's a tensor
            if isinstance(document_embeddings, torch.Tensor):
                document_embeddings = document_embeddings.numpy()
            
            # Create or update the FAISS index
            if st.session_state.faiss_index is None:
                dimension = document_embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(document_embeddings)
                st.session_state.faiss_index = index
                st.session_state.embeddings = document_embeddings
            else:
                # Add new embeddings to existing index
                st.session_state.faiss_index.add(document_embeddings)
                if st.session_state.embeddings is not None:
                    st.session_state.embeddings = np.vstack([st.session_state.embeddings, document_embeddings])
                else:
                    st.session_state.embeddings = document_embeddings
        
        return len(st.session_state.knowledge_base)

# Generate RAG response with the mT5 model
def generate_rag_response(query, top_k=5):
    # Embed the query
    query_embedding = st.session_state.encoder.encode(query).reshape(1, -1)
    
    # Search for relevant documents
    D, I = st.session_state.faiss_index.search(query_embedding, k=top_k)
    
    # Retrieve the relevant documents
    retrieved_documents = [st.session_state.knowledge_base[i] for i in I[0] if i < len(st.session_state.knowledge_base)]
    
    # Format the context for the LLM
    context = "Context: " + " ".join(retrieved_documents) + "\n\nQuestion: " + query + "\n\nAnswer:"
    
    # Generate the response
    input_ids = st.session_state.tokenizer(context, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    output_ids = st.session_state.model.generate(
        input_ids,
        max_length=150,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    response = st.session_state.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Return both the response and the retrieved documents (for transparency)
    return response, retrieved_documents

# HTML templates for UI components
def render_chat_interface():
    # Create a container for the app layout
    st.markdown("""
    <div class="wrapper">
        <div class="chat-app-container">
            <!-- Chat content here -->
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar content - Knowledge Base Management
    with st.sidebar:
        st.markdown("""
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <i class="fas fa-robot"></i> RAG Assistant
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize models if not already loaded
        if st.session_state.tokenizer is None:
            with st.spinner("Loading models... This may take a minute."):
                st.session_state.tokenizer, st.session_state.model, st.session_state.encoder = load_models()
            st.markdown('<div class="alert alert-success">Models loaded successfully!</div>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Select files", 
                                         type=["pdf", "txt", "xlsx", "xls", "json"], 
                                         accept_multiple_files=True)
        
        # Process uploaded files
        if uploaded_files:
            process_button = st.button("Process Files", key="process_files")
            if process_button:
                with st.spinner("Processing files..."):
                    new_documents = []
                    for uploaded_file in uploaded_files:
                        try:
                            if uploaded_file.type == "application/pdf":
                                text = extract_text_from_pdf(uploaded_file)
                            elif uploaded_file.type == "text/plain":
                                text = extract_text_from_txt(uploaded_file)
                            elif uploaded_file.type in ["application/vnd.ms-excel", 
                                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                                text = extract_text_from_excel(uploaded_file)
                            elif uploaded_file.type == "application/json":
                                text = extract_text_from_json(uploaded_file)
                            else:
                                # Try to infer the file type from extension
                                if uploaded_file.name.endswith('.json'):
                                    text = extract_text_from_json(uploaded_file)
                                elif uploaded_file.name.endswith('.pdf'):
                                    text = extract_text_from_pdf(uploaded_file)
                                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                                    text = extract_text_from_excel(uploaded_file)
                                elif uploaded_file.name.endswith('.txt'):
                                    text = extract_text_from_txt(uploaded_file)
                                else:
                                    st.warning(f"Unsupported file type for {uploaded_file.name}. Skipping.")
                                    continue
                            
                            processed_text = preprocess_text(text)
                            new_documents.extend(processed_text)
                            st.success(f"Processed {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Add to knowledge base
                    if new_documents:
                        st.session_state.knowledge_base.extend(new_documents)
                        count = update_knowledge_base()
                        st.success(f"Knowledge base updated with {len(new_documents)} new sentences!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Direct text input for knowledge base
        st.markdown('<div class="section-title">Add Text Directly</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        text_input = st.text_area("Paste text to add", height=100)
        if st.button("Add Text", key="add_text"):
            if text_input:
                processed_text = preprocess_text(text_input)
                st.session_state.knowledge_base.extend(processed_text)
                count = update_knowledge_base()
                st.success(f"Added {len(processed_text)} sentences to knowledge base!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Knowledge base status
        st.markdown('<div class="section-title">Knowledge Base Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="kb-section">', unsafe_allow_html=True)
        
        if st.session_state.knowledge_base:
            st.info(f"Total sentences: {len(st.session_state.knowledge_base)}")
            
            # Option to view knowledge base
            if st.button("View Sample", key="view_kb"):
                sample_size = min(5, len(st.session_state.knowledge_base))
                st.write("Sample from knowledge base:")
                for i in range(sample_size):
                    st.markdown(f"**{i+1}.** {st.session_state.knowledge_base[i][:100]}...", unsafe_allow_html=True)
                    
            # Option to clear knowledge base
            if st.button("Clear All", key="clear_kb"):
                st.session_state.knowledge_base = []
                st.session_state.embeddings = None
                st.session_state.faiss_index = None
                st.success("Knowledge base cleared!")
        else:
            st.warning("Knowledge base is empty")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    st.markdown("""
    <div class="main-content">
        <div class="chat-header">
            <div class="avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div>
                <h2>RAG Assistant</h2>
            </div>
        </div>
        <div class="chat-container" id="chat-container">
    """, unsafe_allow_html=True)
    
    # Display date divider
    st.markdown('<div class="date-divider"><span>Today</span></div>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="message-group user-message-group">
                <div class="message-content">
                    <div class="message-bubble user-message">
                        {message['content']}
                    </div>
                </div>
                <div class="avatar user-avatar">
                    <i class="fas fa-user"></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-group bot-message-group">
                <div class="avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="sender-name">RAG Assistant</div>
                    <div class="message-bubble bot-message">
                        {message['content']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources for bot messages if available
            if 'sources' in message and message['sources']:
                with st.expander("View sources", expanded=False):
                    for i, source in enumerate(message['sources']):
                        st.markdown(f"**Source {i+1}:** {source}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Message input area
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input("", placeholder="Type a message...", key="chat_input", label_visibility="collapsed")
        
    with col2:
        send_pressed = st.button("Send", key="send_button")
    
    # Process user message
    if send_pressed and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate bot response
        if not st.session_state.knowledge_base or st.session_state.faiss_index is None:
            response = "Please add some documents or text to the knowledge base first."
            sources = []
        else:
            with st.spinner("Thinking..."):
                response, sources = generate_rag_response(user_input)
                st.session_state.last_response_sources = sources
        
        # Add bot response with sources
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })
        
        # Rerun to update UI
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Extra components for the chat interface
def inject_js_for_chat_functionality():
    st.markdown("""
    <script>
    
    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    
    document.addEventListener('DOMContentLoaded', function() {
        scrollToBottom();
        
        
        const inputField = document.querySelector('input[data-testid="stTextInput"]');
        if (inputField) {
            inputField.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    // Find and click the send button
                    const sendButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                    if (sendButton) {
                        sendButton.click();
                    }
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Main function to run the app
def main():
    # Add font-awesome script
    st.markdown("""
    <script src="https://kit.fontawesome.com/4ac40a47d1.js" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)
    
    # Render the chat interface
    render_chat_interface()
    
    # Add JavaScript functionality
    inject_js_for_chat_functionality()

if __name__ == "__main__":
    main()