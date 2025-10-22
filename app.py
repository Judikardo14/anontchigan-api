import json
import os
import sys
import logging
from typing import Dict, List, Optional
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from datetime import datetime
import time

# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANONTCHIGAN")

class Config:
    """Configuration optimisée"""
    SIMILARITY_THRESHOLD = 0.75
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600
    FAISS_RESULTS_COUNT = 3
    MIN_ANSWER_LENGTH = 30

# ============================================
# CSS PERSONNALISÉ - BOT PROFESSIONNEL
# ============================================

st.markdown("""
<style>
    /* MASQUER TOUS LES ÉLÉMENTS STREAMLIT */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    .stAppHeader {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    div[data-testid="stToolbar"] > div {display: none !important;}
    .styles_viewerBadge__1yB5_ {display: none !important;}
    
    /* VARIABLES CSS */
    :root {
        --rose-primary: #E91E63;
        --rose-dark: #C2185B;
        --violet: #9C27B0;
        --blanc: #FFFFFF;
        --gris-clair: #F5F5F5;
        --gris-message: #F0F0F0;
        --ombre: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* BACKGROUND GLOBAL */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        overflow-x: hidden;
    }
    
    /* CONTENEUR PRINCIPAL */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* HEADER DU BOT */
    .bot-header {
        background: white;
        padding: 1rem 1.5rem;
        box-shadow: var(--ombre);
        display: flex;
        align-items: center;
        gap: 1rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .bot-avatar {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, var(--rose-primary), var(--violet));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        position: relative;
    }
    
    .status-dot {
        width: 12px;
        height: 12px;
        background: #4CAF50;
        border: 2px solid white;
        border-radius: 50%;
        position: absolute;
        bottom: 2px;
        right: 2px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .bot-info h2 {
        margin: 0;
        font-size: 1.2rem;
        color: #333;
    }
    
    .bot-info p {
        margin: 0;
        font-size: 0.85rem;
        color: #4CAF50;
    }
    
    /* ZONE DE CHAT */
    .chat-container {
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 2rem 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    /* MESSAGES UTILISATEUR */
    .user-message {
        background: linear-gradient(135deg, var(--rose-primary), var(--violet));
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        max-width: 70%;
        box-shadow: var(--ombre);
        animation: slideInRight 0.3s ease;
        word-wrap: break-word;
    }
    
    /* MESSAGES BOT */
    .bot-message {
        background: var(--gris-message);
        color: #333;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 70%;
        box-shadow: var(--ombre);
        animation: slideInLeft 0.3s ease;
        word-wrap: break-word;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* TYPING INDICATOR */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem 1.2rem;
        background: var(--gris-message);
        border-radius: 18px;
        max-width: 80px;
        margin-right: auto;
        box-shadow: var(--ombre);
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #666;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* QUESTIONS RAPIDES */
    .quick-questions {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        padding: 1rem 1.5rem;
        animation: fadeIn 0.5s ease;
    }
    
    .quick-question-btn {
        background: white;
        border: 2px solid var(--rose-primary);
        color: var(--rose-primary);
        padding: 0.9rem 1.2rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        text-align: left;
        box-shadow: var(--ombre);
    }
    
    .quick-question-btn:hover {
        background: linear-gradient(135deg, var(--rose-primary), var(--violet));
        color: white;
        transform: translateX(5px);
        border-color: var(--violet);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* INPUT ZONE */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem 1.5rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 100;
    }
    
    .stTextInput input {
        border-radius: 25px !important;
        border: 2px solid var(--rose-primary) !important;
        padding: 0.9rem 1.5rem !important;
        font-size: 1rem !important;
        width: 100% !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--violet) !important;
        box-shadow: 0 0 0 3px rgba(233,30,99,0.15) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: #999 !important;
        font-style: italic;
    }
    
    /* BUTTON STYLING */
    .stButton button {
        background: linear-gradient(135deg, var(--rose-primary), var(--violet)) !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.9rem 2rem !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 16px rgba(233,30,99,0.4) !important;
    }
    
    /* BADGES SOURCE */
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
        opacity: 0.7;
        font-weight: 500;
    }
    
    .source-badge.json_direct {
        background: #E8F5E8;
        color: #2E7D32;
    }
    
    .source-badge.groq_generated {
        background: #E3F2FD;
        color: #1565C0;
    }
    
    .source-badge.salutation {
        background: #F3E5F5;
        color: #7B1FA2;
    }
    
    /* SCROLLBAR */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(233,30,99,0.3);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: rgba(233,30,99,0.5);
    }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 85%;
        }
        
        .chat-container {
            height: calc(100vh - 220px);
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SERVICE GROQ
# ============================================

class GroqService:
    def __init__(self):
        self.client = None
        self.available = False
        self._initialize_groq()
    
    def _initialize_groq(self):
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY", "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")
            if not api_key:
                logger.warning("Clé API Groq manquante")
                return
            
            self.client = Groq(api_key=api_key)
            
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.1-8b-instant",
                max_tokens=5,
            )
            self.available = True
            logger.info("✓ Service Groq initialisé")
            
        except Exception as e:
            logger.warning(f"Service Groq non disponible: {str(e)}")
    
    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        
        try:
            context_short = self._prepare_context(context)
            messages = self._prepare_messages(question, context_short, history)
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                top_p=0.9,
            )
            
            answer = response.choices[0].message.content.strip()
            answer = self._clean_response(answer)
            
            if not self._is_valid_answer(answer):
                raise ValueError("Réponse trop courte")
                
            answer = self._ensure_complete_response(answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Erreur Groq: {str(e)}")
            raise
    
    def _prepare_context(self, context: str) -> str:
        lines = context.split('\n')[:5]
        context_short = '\n'.join(lines)
        if len(context_short) > Config.MAX_CONTEXT_LENGTH:
            context_short = context_short[:Config.MAX_CONTEXT_LENGTH-3] + "..."
        return context_short
    
    def _prepare_messages(self, question: str, context: str, history: List[Dict]) -> List[Dict]:
        system_prompt = f"""Tu es ANONTCHIGAN, assistante IA professionnelle spécialisée dans la sensibilisation au cancer du sein au Bénin.

CONTEXTE À UTILISER :
{context}

RÈGLES CRITIQUES :
1. FOURNIR DES RÉPONSES COMPLÈTES - ne JAMAIS couper une phrase en milieu de mot
2. Tes réponses doivent se terminer naturellement par un point final
3. Tes créateurs sont Judicaël Karol DOBOEVI, Ursus Hornel GBAGUIDI, Abel Kokou KPOCOUTA et Josaphat ADJELE

STYLE :
- Professionnel, clair, empathique
- Réponses directes sans formules introductives
- CONCIS mais COMPLET
- Emojis : 💗 🌸 😊 🇧🇯"""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history[-4:]:
            messages.append(msg)
        
        messages.append({
            "role": "user", 
            "content": f"QUESTION: {question}\n\nIMPORTANT : Réponds de façon COMPLÈTE sans couper ta réponse."
        })
        
        return messages
    
    def _clean_response(self, answer: str) -> str:
        unwanted_intros = [
            'bonjour', 'salut', 'coucou', 'hello', 'akwè', 'yo', 'bonsoir', 'hi',
            'excellente question', 'je suis ravi', 'permettez-moi'
        ]
        
        answer_lower = answer.lower()
        for phrase in unwanted_intros:
            if answer_lower.startswith(phrase):
                sentences = answer.split('.')
                if len(sentences) > 1:
                    answer = '.'.join(sentences[1:]).strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:]
                break
        
        return answer.strip()
    
    def _is_valid_answer(self, answer: str) -> bool:
        return len(answer) >= Config.MIN_ANSWER_LENGTH
    
    def _ensure_complete_response(self, answer: str) -> str:
        if not answer:
            return answer
            
        if answer.endswith(('...', ',', ';', ' ')):
            last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
            
            if last_period > 0:
                answer = answer[:last_period + 1]
            else:
                answer = answer.rstrip(' ,;...')
                if not answer.endswith(('.', '!', '?')):
                    answer += '.'
        
        return answer

# ============================================
# SERVICE RAG
# ============================================

class RAGService:
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self._load_data(data_file)
        self._initialize_embeddings()
    
    def _load_data(self, data_file: str):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                self.questions_data.append({
                    'question_originale': item['question'],
                    'question_normalisee': item['question'].lower().strip(),
                    'answer': item['answer']
                })
            
            logger.info(f"✓ {len(self.questions_data)} questions chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        try:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            all_texts = [
                f"Q: {item['question_originale']} R: {item['answer']}"
                for item in self.questions_data
            ]
            
            self.embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
            self.embeddings = np.array(self.embeddings).astype('float32')
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
            logger.info(f"✓ Index FAISS créé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation embeddings: {str(e)}")
            raise
    
    def search(self, query: str, k: int = Config.FAISS_RESULTS_COUNT) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.questions_data):
                    similarity = 1 / (1 + distances[0][i])
                    results.append({
                        'question': self.questions_data[idx]['question_originale'],
                        'answer': self.questions_data[idx]['answer'],
                        'similarity': similarity,
                        'distance': distances[0][i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche FAISS: {str(e)}")
            return []

# ============================================
# FONCTION DE TRAITEMENT DES QUESTIONS
# ============================================

def process_question(question: str, history: List[Dict], groq_service, rag_service):
    salutations = ["cc", "bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
    question_lower = question.lower().strip()
    
    if any(salut == question_lower for salut in salutations):
        responses = [
            "Je suis ANONTCHIGAN, assistante pour la sensibilisation au cancer du sein. Comment puis-je vous aider ? 💗",
            "Bonjour ! Je suis ANONTCHIGAN. Que souhaitez-vous savoir sur le cancer du sein ? 🌸",
            "ANONTCHIGAN à votre service. Posez-moi vos questions sur la prévention du cancer du sein. 😊"
        ]
        return {
            "answer": random.choice(responses),
            "method": "salutation"
        }
    
    faiss_results = rag_service.search(question)
    
    if not faiss_results:
        return {
            "answer": "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin. 💗",
            "method": "no_result"
        }
    
    best_result = faiss_results[0]
    similarity = best_result['similarity']
    
    if similarity >= Config.SIMILARITY_THRESHOLD:
        return {
            "answer": best_result['answer'],
            "method": "json_direct"
        }
    else:
        context_parts = []
        for i, result in enumerate(faiss_results[:3], 1):
            answer_truncated = result['answer'][:200] + "..."
            context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
        
        context = "\n\n".join(context_parts)
        
        try:
            if groq_service.available:
                answer = groq_service.generate_response(question, context, history)
                method = "groq_generated"
            else:
                answer = "Je vous recommande de consulter un professionnel de santé. 💗"
                method = "fallback"
        except:
            answer = "Pour des informations précises, veuillez consulter un médecin. 🌸"
            method = "error_fallback"
        
        return {
            "answer": answer,
            "method": method
        }

# ============================================
# INITIALISATION
# ============================================

@st.cache_resource
def load_services():
    groq = GroqService()
    rag = RAGService()
    return groq, rag

# ============================================
# INTERFACE
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

groq_service, rag_service = load_services()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_quick_questions" not in st.session_state:
    st.session_state.show_quick_questions = True
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

# Header
st.markdown("""
<div class="bot-header">
    <div class="bot-avatar">
        💗
        <div class="status-dot"></div>
    </div>
    <div class="bot-info">
        <h2>ANONTCHIGAN</h2>
        <p>🟢 En ligne - Statut actif</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Zone de chat
chat_placeholder = st.container()

with chat_placeholder:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Questions rapides
    if st.session_state.show_quick_questions and len(st.session_state.messages) == 0:
        st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">💬 Questions rapides :</p>', unsafe_allow_html=True)
        
        quick_questions = [
            "Quels sont les symptômes du cancer du sein ?",
            "Comment faire l'autopalpation ?",
            "Quels sont les facteurs de risque ?",
            "Où se faire dépister au Bénin ?"
        ]
        
        cols = st.columns(1)
        for i, q in enumerate(quick_questions):
            if cols[0].button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.show_quick_questions = False
                st.session_state.is_typing = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            method = message.get("method", "")
            badge = f'<span class="source-badge {method}">{method.replace("_", " ")}</span>'
            st.markdown(f'<div class="bot-message">{message["content"]}<br>{badge}</div>', unsafe_allow_html=True)
    
    # Typing indicator
    if st.session_state.is_typing:
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input zone
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Posez-moi une question...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("💬", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if send_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.show_quick_questions = False
    st.session_state.is_typing = True
    st.rerun()

if st.session_state.is_typing and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "user":
        time.sleep(1)
        result = process_question(
            st.session_state.messages[-1]["content"],
            [],
            groq_service,
            rag_service
        )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "method": result["method"]
        })
        st.session_state.is_typing = False
        st.rerun()
