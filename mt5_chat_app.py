import streamlit as st

# Streamlit Page Configuration - ต้องเป็นคำสั่ง Streamlit แรกในสคริปต์
st.set_page_config(
    page_title="🏠 AI Property Consultant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import faiss
import re
import torch
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer

# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.5s;
        border: 1px solid #D4AF37;
    }
    .user-message {
        background-color: #FFF8E7;
        border-left: 5px solid #D4AF37;
    }
    .assistant-message {
        background-color: #FFFAF0;
        border-left: 5px solid #B8860B;
    }
    .property-card {
        background-color: #FFF8E7;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(212, 175, 55, 0.2);
        transition: transform 0.3s ease;
        border: 1px solid #D4AF37;
    }
    .property-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(212, 175, 55, 0.3);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .thinking {
        animation: pulse 1.5s infinite;
        color: #D4AF37;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        background-color: #D4AF37;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #B8860B;
        transform: translateY(-2px);
    }
    .stSelectbox select {
        border-color: #D4AF37;
    }
    .stSelectbox select:focus {
        border-color: #B8860B;
    }
    .stTextInput input {
        border-color: #D4AF37;
    }
    .stTextInput input:focus {
        border-color: #B8860B;
    }
    h1, h2, h3 {
        color: #B8860B;
    }
    .stMarkdown {
        color: #4A4A4A;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the models for faster loading
@st.cache_resource
def load_language_model(model_name):
    try:
        # Use the specific T5 tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, force_download=True)
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

class RealEstateConsultationAI:
    def __init__(self, verbose: bool = False):
        """
        Initialize an advanced real estate consultation AI system
        
        Args:
            verbose (bool): Enable detailed logging and diagnostics
        """
        # Logging and Diagnostics
        self.verbose = verbose
        
        # Initialize Models
        try:
            with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
                # Multilingual Embedding Model
                embedding_model_name = 'paraphrase-multilingual-mpnet-base-v2'
                self.embedding_model = load_embedding_model(embedding_model_name)
                
                # Language Generation Model
                self.language_model_name = 'google/mt5-base'
                self.tokenizer, self.language_model = load_language_model(self.language_model_name)
                
                # Core Data Management
                self.vector_store = None
                self.document_store = []
                
                # Retrieval Configuration
                self.retrieval_config = {
                    'top_k': 5,
                    'similarity_threshold': 0.3,
                    'context_window': 3
                }
                
                # Consultation Styles
                self.consultation_styles = {
                    'ทางการ': {
                        'tone': 'Formal real estate consultation',
                        'temperature': 0.5,
                        'guidance': 'Provide precise, fact-based property insights.'
                    },
                    'เป็นกันเอง': {
                        'tone': 'Personalized, approachable advisor',
                        'temperature': 0.7,
                        'guidance': 'Offer warm, tailored property recommendations.'
                    }
                }
            
            st.success("🚀 โหลดโมเดล AI เรียบร้อยแล้ว")
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing with semantic preservation
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Cleaned, normalized text
        """
        try:
            # Comprehensive text cleaning
            text = text.lower()
            # Preserve Thai, English, numbers, and key punctuation
            text = re.sub(r'[^\w\s\u0E00-\u0E7F0-9]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        except Exception as e:
            st.warning(f"เกิดข้อผิดพลาดในการประมวลผลข้อความ: {e}")
            return text
    
    def create_vector_database(self, dataframe: pd.DataFrame):
        """
        Create a semantic vector database from property data
        
        Args:
            dataframe (pd.DataFrame): Property dataset
        """
        try:
            with st.spinner("🗂️ กำลังสร้างฐานข้อมูลอสังหาริมทรัพย์..."):
                # Generate searchable text representations
                searchable_texts = dataframe.apply(
                    lambda row: self.preprocess_text(
                        ' '.join([
                            str(val) for val in row 
                            if pd.notna(val)
                        ])
                    ),
                    axis=1
                )
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(
                    searchable_texts.tolist(), 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.vector_store = faiss.IndexFlatL2(dimension)
                
                # Normalize embeddings
                faiss.normalize_L2(embeddings)
                
                # Add embeddings to vector store
                self.vector_store.add(embeddings.astype('float32'))
                
                # Store original documents
                self.document_store = dataframe.to_dict('records')
            
            st.success(f"🏠 สร้างฐานข้อมูลสำเร็จ มีอสังหาริมทรัพย์ {len(self.document_store)} รายการ")
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้างฐานข้อมูล: {e}")
            raise
    
    def semantic_search(self, query: str) -> List[Dict]:
        """
        Perform advanced semantic search on property database
        
        Args:
            query (str): User's search query
        
        Returns:
            List[Dict]: Relevant property recommendations
        """
        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([processed_query])
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Perform similarity search
            distances, indices = self.vector_store.search(
                query_embedding.astype('float32'), 
                self.retrieval_config['top_k']
            )
            
            # Filter and rank results
            relevant_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_store):  # Safety check
                    similarity = 1 / (1 + distances[0][i])
                    
                    if similarity > self.retrieval_config['similarity_threshold']:
                        document = self.document_store[idx]
                        document['similarity_score'] = similarity
                        relevant_results.append(document)
            
            # Sort by similarity score
            return sorted(
                relevant_results, 
                key=lambda x: x['similarity_score'], 
                reverse=True
            )
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการค้นหา: {e}")
            return []
    
    def generate_consultation_response(
        self, 
        query: str, 
        results: List[Dict], 
        consultation_style: str = 'ทางการ'
    ) -> str:
        """
        Generate contextually rich consultation response
        
        Args:
            query (str): User's input query
            results (List[Dict]): Semantic search results
            consultation_style (str): Response generation style
        
        Returns:
            str: Generated consultation response
        """
        try:
            # Select style configuration
            style_config = self.consultation_styles.get(
                consultation_style, 
                self.consultation_styles['ทางการ']
            )
            
            # Construct comprehensive context
            context = "\n".join([
                f"***{result.get('โครงการ', 'ไม่มีข้อมูล')} | "
                f"แถว: {result.get('ตำแหน่ง', 'ไม่มีข้อมูล')} | "
                f"ทำเล: {result.get('ตำแหน่ง', 'ไม่มีข้อมูล')} | "
                f"อยู่: {result.get('ตำแหน่ง', 'ไม่มีข้อมูล')} | "
                f"ราคา: {result.get('ราคา', 'ไม่มีข้อมูล')} | "
                f"ติด: {result.get('สถานีรถไฟฟ้า', 'ไม่มีข้อมูล')} | "
                f"ใกล้: {result.get('สถานีรถไฟฟ้า', 'ไม่มีข้อมูล')} | "
                f"ใกล้: {result.get('โรงพยาบาล', 'ไม่มีข้อมูล')} | "
                f"ใกล้: {result.get('ห้างสรรพสินค้า', 'ไม่มีข้อมูล')} | "
                f"ติด: {result.get('โรงพยาบาล', 'ไม่มีข้อมูล')} | "
                f"ติด: {result.get('ห้างสรรพสินค้า', 'ไม่มีข้อมูล')} | "
                f"แบบ: {result.get('รูปแบบ', 'ไม่มีข้อมูล')} | "
                f"หา: {result.get('ประเภท', 'ไม่มีข้อมูล')} | "
                f"ใกล้: {result.get('สนามบิน', 'ไม่มีข้อมูล')} | "
                f"ใกล้: {result.get('สถานศึกษา', 'ไม่มีข้อมูล')}"

                for result in results[:3]
            ])
            
            # Prepare generation prompt
            prompt = f"""บริบท: {context}
คำถาม: {query}
คำแนะนำ: {style_config['guidance']}
คำตอบ:"""
            
            # Tokenize and generate response
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            outputs = self.language_model.generate(
                **inputs,
                max_length=300,
                num_beams=8,
                temperature=style_config['temperature'],
                do_sample=True
            )
            
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return response
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}")
            return "ขออภัย ไม่สามารถให้คำปรึกษาได้ในขณะนี้"

def clear_chat_history():
    st.session_state.messages = []
    st.rerun()

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Set sidebar state based on previous toggle if available
    sidebar_state = 'expanded'
    if 'sidebar_state_next' in st.session_state:
        sidebar_state = st.session_state.sidebar_state_next
        # Remove from session state to avoid infinite loop
        del st.session_state.sidebar_state_next
        st.rerun()  # Apply the new sidebar state
    
    # Initialize Real Estate AI
    if 'real_estate_ai' not in st.session_state:
        st.session_state.real_estate_ai = RealEstateConsultationAI(verbose=True)
    real_estate_ai = st.session_state.real_estate_ai
    
    # Page Title
    # Add logo
    st.image(
        "https://github.com/Phattarapong26/image/blob/main/Screenshot%202568-03-28%20at%2020.38.51.png?raw=true",
        width=40,
        use_container_width=True
    )
    st.title("Fundee finding Real Estate")
    
    # Sidebar for File Upload and Configuration
    with st.sidebar:
        st.header("🔧 จัดการระบบ")
        
        # Property Data Upload
        uploaded_file = st.file_uploader(
            "อัพโหลดไฟล์ข้อมูลอสังหาริมทรัพย์", 
            type=['xlsx', 'csv']
        )
        
        # Consultation Style Selection
        consultation_style = st.selectbox(
            "รูปแบบการให้คำปรึกษา", 
            list(real_estate_ai.consultation_styles.keys())
        )
        
        # Add clear chat history button
        st.button("🗑️ ล้างประวัติการแชท", on_click=clear_chat_history)
    
    # Main Chat Interface
    main_container = st.container()
    
    if uploaded_file is not None:
        try:
            # Check if we already processed this file
            if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
                # Load Dataset
                with st.spinner("📊 กำลังโหลดข้อมูล..."):
                    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
                
                # Required Columns Validation
                required_columns = [
                    'โครงการ', 'ตำแหน่ง', 'ราคา', 'ประเภท', 'รูปแบบ', 
                    'สถานศึกษา', 'ห้างสรรพสินค้า', 'สถานีรถไฟฟ้า', 
                    'โรงพยาบาล', 'สนามบิน'
                ]
                
                if not all(col in df.columns for col in required_columns):
                    st.error("ไฟล์ข้อมูลไม่มีคอลัมน์ที่จำเป็น โปรดตรวจสอบรูปแบบไฟล์")
                    return
                
                # Create Semantic Vector Database
                real_estate_ai.create_vector_database(df)
                
                # Save current file name to session state
                st.session_state.current_file = uploaded_file.name
            
            # Initialize or Load Chat History
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Chat message container
            chat_container = st.container()
            
            with chat_container:
                # Display Chat Messages with improved styling
                for message in st.session_state.messages:
                    role_class = "user-message" if message["role"] == "user" else "assistant-message"
                    role_label = "👤 คุณ" if message["role"] == "user" else "🤖 AI"
                    st.markdown(
                        f"<div class='chat-message {role_class}'><strong>{role_label}:</strong> {message['content']}</div>", 
                        unsafe_allow_html=True
                    )
            
            # Chat Input
            prompt = st.chat_input("คุณกำลังมองหาอสังหาริมทรัพย์แบบไหน?")
            
            if prompt:
                # Add User Message to Chat History
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                
                # Display User Message (will be shown in next rerun)
                with chat_container:
                    st.markdown(
                        f"<div class='chat-message user-message'><strong>👤 คุณ:</strong> {prompt}</div>", 
                        unsafe_allow_html=True
                    )
                
                # Show thinking animation
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("<div class='thinking'>🧠 กำลังคิด...</div>", unsafe_allow_html=True)
                
                # Perform Semantic Search
                search_results = real_estate_ai.semantic_search(prompt)
                
                # Generate AI Response
                if search_results:
                    # ตรวจสอบคะแนนความคล้ายคลึง
                    max_similarity = max(result['similarity_score'] for result in search_results)
                    
                    if max_similarity < 0.5:  # ถ้าคะแนนสูงสุดต่ำกว่า 0.5
                        low_score_response = (
                            "ฉันพยายามค้นหาอสังหาที่น่าจะตรงกับที่คุณสนใจไม่เจอ"                           
                            "เพิ่มรายละเอียดจากเดิมได้ไหม เดี๋ยวฉันจะช่วยหาอีกครั้ง"
                        )
                        
                        with chat_container:
                            st.markdown(
                                f"<div class='chat-message assistant-message'>{low_score_response}</div>", 
                                unsafe_allow_html=True
                            )
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": low_score_response
                        })
                    else:
                        with st.spinner("🤖 กำลังสร้างคำตอบ..."):
                            ai_response = real_estate_ai.generate_consultation_response(
                                prompt, 
                                search_results, 
                                consultation_style
                            )
                        
                        # Clear thinking animation
                        thinking_placeholder.empty()
                        
                        # Display AI Response with improved styling
                        with chat_container:
                            st.markdown(
                                f"<div class='chat-message assistant-message'><strong>🤖 AI:</strong> {ai_response}</div>", 
                                unsafe_allow_html=True
                            )
                        
                        # Add AI Message to Chat History
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ai_response
                        })
                        
                        # Display Recommended Properties after chat
                        st.subheader("อสังหาริมทรัพย์ที่น่าจะตรงกับที่คุณสนใจ")
                        
                        for i, result in enumerate(search_results[:2]):
                            st.markdown(
                                f"""
                                ### {result['โครงการ']}
                                
                                **ทำเล:** {result['ตำแหน่ง']}  
                                **ราคา:** {result['ราคา']}  
                                **ประเภท:** {result['ประเภท']}  
                                **รูปแบบ:** {result['รูปแบบ']}  
                                
                                **ข้อมูลเพิ่มเติม:**
                                - ใกล้รถไฟฟ้า: {result['สถานีรถไฟฟ้า']}
                                - ใกล้โรงพยาบาล: {result['โรงพยาบาล']}
                                - ใกล้สถานศึกษา: {result['สถานศึกษา']}
                                - ใกล้ห้างสรรพสินค้า: {result['ห้างสรรพสินค้า']}
                                - ใกล้สนามบิน: {result['สนามบิน']}
                                
                                ---
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    # Clear thinking animation
                    thinking_placeholder.empty()
                    
                    # Handle No Results
                    no_results_response = (
                        "ขออภัย ไม่พบอสังหาริมทรัพย์ที่ตรงกับความต้องการของคุณ "
                        "กรุณาลองใช้คำค้นหาอื่นหรือระบุความต้องการให้กว้างขึ้น"
                    )
                    
                    with chat_container:
                        st.markdown(
                            f"<div class='chat-message assistant-message'><strong>🤖 AI:</strong> {no_results_response}</div>", 
                            unsafe_allow_html=True
                        )
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": no_results_response
                    })
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
    else:
        st.info("👈 กรุณาอัพโหลดไฟล์ข้อมูลอสังหาริมทรัพย์ที่เมนูด้านซ้าย เพื่อเริ่มการสนทนา")

if __name__ == "__main__":
    main()
