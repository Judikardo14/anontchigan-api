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
# CSS PERSONNALISÉ - INTERFACE MODERNE
# ============================================

st.markdown("""
<style>
    /* MASQUER ABSOLUMENT TOUS LES ÉLÉMENTS STREAMLIT */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    .stAppHeader {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    button[title="Manage app"] {display: none !important;}
    div[data-testid="stToolbar"] > div {display: none !important;}
    div[data-testid="manage-app-button"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    div.styles_viewerBadge__1yB5_ {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    .viewerBadge_link__1S137 {display: none !important;}
    .viewerBadge_text__1JaDK {display: none !important;}
    
    /* VARIABLES CSS - OCTOBRE ROSE */
    :root {
        --rose-primary: #E91E63;
        --rose-clair: #FCE4EC;
        --rose-fonce: #C2185B;
        --blanc: #FFFFFF;
        --gris-clair: #F8F9FA;
        --beige-clair: #FFF3E0;
        --ombre-douce: 0 2px 12px rgba(233, 30, 99, 0.15);
    }
    
    /* BACKGROUND GLOBAL - BLANC PUR */
    .stApp {
        background: #FFFFFF !important;
    }
    
    /* CONTENEUR PRINCIPAL */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        background: white !important;
    }
    
    /* NAVBAR SUPÉRIEURE - ROSE OCTOBRE */
    .top-navbar {
        background: linear-gradient(135deg, #E91E63 0%, #D81B60 100%);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.25);
    }
    
    .navbar-logo {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: white;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 0.5px;
    }
    
    .navbar-menu {
        display: flex;
        gap: 2rem;
    }
    
    .navbar-menu a {
        color: white;
        text-decoration: none;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .navbar-menu a:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
    }
    
    .navbar-menu a.active {
        background: rgba(255,255,255,0.3);
    }
    
    /* HEADER AVEC AVATAR - DESIGN ÉLÉGANT */
    .bot-header-card {
        background: linear-gradient(135deg, #E91E63 0%, #F06292 100%);
        padding: 2rem 2.5rem;
        margin: 0;
        border-radius: 0 0 40px 40px;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 6px 25px rgba(233, 30, 99, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .bot-header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .bot-avatar-circle {
        width: 65px;
        height: 65px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        position: relative;
        z-index: 2;
    }
    
    .bot-header-text {
        position: relative;
        z-index: 2;
    }
    
    .bot-header-text h2 {
        margin: 0;
        color: white;
        font-size: 1.6rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-header-text p {
        margin: 0.3rem 0 0 0;
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* BANNIÈRE BIENVENUE - DESIGN PREMIUM */
    .welcome-banner {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        margin: 2rem 2.5rem 1.5rem 2.5rem;
        padding: 3rem 2rem;
        border-radius: 30px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(233, 30, 99, 0.15);
        border: 2px solid rgba(233, 30, 99, 0.1);
    }
    
    .welcome-banner .ribbon-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 8px rgba(233, 30, 99, 0.3));
    }
    
    .welcome-banner h3 {
        color: #C2185B;
        font-size: 1.8rem;
        margin: 1rem 0;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    
    .welcome-banner p {
        color: #880E4F;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        line-height: 1.6;
    }
    
    /* ZONE DE CHAT - FOND BLANC */
    .chat-container {
        padding: 1.5rem 2.5rem;
        max-height: calc(100vh - 450px);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        background: white;
    }
    
    /* MESSAGE BOT AVEC AVATAR - DESIGN SOIGNÉ */
    .bot-message-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: slideInLeft 0.4s ease-out;
    }
    
    .bot-mini-avatar {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #E91E63, #F06292);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
        box-shadow: 0 3px 10px rgba(233, 30, 99, 0.3);
    }
    
    .bot-message-content {
        background: #FFFFFF;
        color: #333;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 75%;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.08);
        word-wrap: break-word;
        line-height: 1.7;
        border: 1px solid rgba(233, 30, 99, 0.1);
    }
    
    .bot-message-content ul {
        margin: 0.8rem 0;
        padding-left: 1.8rem;
    }
    
    .bot-message-content li {
        margin: 0.5rem 0;
        color: #444;
    }
    
    /* ENCADRÉ D'AVERTISSEMENT - ROSE */
    .info-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 5px solid #E91E63;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-radius: 12px;
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        box-shadow: 0 2px 8px rgba(233, 30, 99, 0.1);
    }
    
    .info-box-icon {
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    
    .info-box-text {
        font-size: 0.95rem;
        color: #5D4037;
        line-height: 1.6;
    }
    
    .info-box-text strong {
        color: #E91E63;
        font-weight: 700;
    }
    
    /* MESSAGES UTILISATEUR - ROSE ÉLÉGANT */
    .user-message {
        background: linear-gradient(135deg, #E91E63 0%, #F06292 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin-left: auto;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3);
        animation: slideInRight 0.4s ease-out;
        word-wrap: break-word;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* TYPING INDICATOR - ROSE */
    .typing-wrapper {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 3px 15px rgba(233, 30, 99, 0.15);
        border: 1px solid rgba(233, 30, 99, 0.1);
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        background: #E91E63;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
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
            opacity: 0.4;
        }
        30% {
            transform: translateY(-12px);
            opacity: 1;
        }
    }
    
    /* QUESTIONS RAPIDES - DESIGN MODERNE */
    .quick-questions-section {
        padding: 0 2.5rem 1.5rem 2.5rem;
        background: white;
    }
    
    .quick-questions-title {
        color: #E91E63;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* BOUTONS QUESTIONS - STYLE PILULES */
    .stButton button {
        background: white !important;
        color: #E91E63 !important;
        border: 2px solid #E91E63 !important;
        border-radius: 25px !important;
        padding: 0.9rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 10px rgba(233, 30, 99, 0.15) !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #E91E63, #F06292) !important;
        color: white !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.3) !important;
        border-color: #F06292 !important;
    }
    
    /* INPUT ZONE FIXE EN BAS - ÉLÉGANTE */
    .input-fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1.2rem 2.5rem;
        box-shadow: 0 -4px 20px rgba(233, 30, 99, 0.1);
        z-index: 1000;
        border-top: 2px solid rgba(233, 30, 99, 0.1);
    }
    
    .input-wrapper {
        display: flex;
        align-items: center;
        gap: 1rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .stTextInput input {
        border-radius: 30px !important;
        border: 2px solid #E91E63 !important;
        padding: 1.1rem 1.8rem !important;
        font-size: 1rem !important;
        width: 100% !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #F06292 !important;
        box-shadow: 0 0 0 4px rgba(233, 30, 99, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: #999 !important;
        font-style: italic;
    }
    
    /* BOUTON D'ENVOI - CIRCULAIRE ROSE */
    div[data-testid="column"]:last-child .stButton button {
        background: linear-gradient(135deg, #E91E63, #F06292) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 55px !important;
        height: 55px !important;
        min-width: 55px !important;
        border: none !important;
        font-size: 1.5rem !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.4) !important;
    }
    
    div[data-testid="column"]:last-child .stButton button:hover {
        transform: scale(1.1) rotate(15deg) !important;
        box-shadow: 0 6px 25px rgba(233, 30, 99, 0.5) !important;
    }
    
    /* SCROLLBAR PERSONNALISÉE - ROSE */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(233, 30, 99, 0.05);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #E91E63, #F06292);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #C2185B, #E91E63);
    }
    
    /* RESPONSIVE - MOBILE */
    @media (max-width: 768px) {
        .navbar-menu {
            display: none;
        }
        
        .bot-message-content,
        .user-message {
            max-width: 85%;
        }
        
        .chat-container {
            max-height: calc(100vh - 480px);
            padding: 1rem 1.5rem;
        }
        
        .welcome-banner {
            padding: 2rem 1.5rem;
            margin: 1.5rem 1.5rem 1rem 1.5rem;
        }
        
        .input-fixed-bottom {
            padding: 1rem 1.5rem;
        }
        
        .bot-header-card {
            padding: 1.5rem 1.5rem;
        }
        
        .navbar-logo {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SERVICE GROQ (NON MODIFIÉ)
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
            
            # Test de connexion
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
        """Génère une réponse complète sans coupure"""
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        
        try:
            context_short = self._prepare_context(context)
            messages = self._prepare_messages(question, context_short, history)
            
            logger.info("🤖 Génération avec Groq...")
            
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
            
            logger.info(f"✓ Réponse générée ({len(answer)} caractères)")
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
2. Si tu donnes un conseil de prévention, aller deux fois à la ligne AVANT de le donner
3. Tes réponses doivent se terminer naturellement par un point final
4. Tes créateurs sont Judicaël Karol DOBOEVI, Ursus Hornel GBAGUIDI, Abel Kokou KPOCOUTA et Josaphat ADJELE, tous membres du club d'IA de l'ENSGMM (Ecole Nationale Supérieure de Génie Mathématique et Modélisation) d'Abomey.

STYLE :
- Professionnel, clair, empathique
- Réponses directes sans formules introductives
- CONCIS mais COMPLET
- Humour humain léger 1 fois sur 3 réponses
- Emojis à utiliser fréquemment : 💗 🌸 😊 🇧🇯

STRUCTURE DES RÉPONSES :
1. Réponse principale basée sur le contexte
2. N'invente PAS d'informations
3. Si le contexte ne couvre pas la question, recommande de consulter un professionnel de santé au Bénin
4. ENSGMM se définit comme l'Ecole Nationale Supérieure de Génie Mathématique et Modélisation
5. Utilise des connecteurs logiques pour fluidifier et produire des réponses naturelles et cohérentes

DIRECTIVES ANTI-COUPURE :
- Vérifie que ta réponse est complète avant de terminer
- Ne coupe PAS en milieu de phrase ou de mot
- Utilise "Atassa!" ou "Atassaaaaa!" en début de phrase en cas d'humour et d'étonnement extrême
- Termine par un point final approprié
- Si tu mentionnes des noms (créateurs, etc.), assure-toi qu'ils sont COMPLETS

Conseils de prévention : seulement si pertinents et si demandés."""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history[-4:]:
            messages.append(msg)
        
        messages.append({
            "role": "user", 
            "content": f"QUESTION: {question}\n\nIMPORTANT : Réponds de façon COMPLÈTE sans couper ta réponse. Termine par un point final. Si conseil de prévention, va à la ligne avant."
        })
        
        return messages
    
    def _clean_response(self, answer: str) -> str:
        unwanted_intros = [
            'bonjour', 'salut', 'coucou', 'hello', 'akwè', 'yo', 'bonsoir', 'hi',
            'excellente question', 'je suis ravi', 'permettez-moi', 'tout d abord',
            'premièrement', 'pour commencer', 'en tant qu', 'je suis anontchigan'
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
        return (len(answer) >= Config.MIN_ANSWER_LENGTH and 
                not answer.lower().startswith(('je ne sais pas', 'désolé', 'sorry')))
    
    def _ensure_complete_response(self, answer: str) -> str:
        if not answer:
            return answer
            
        cut_indicators = [
            answer.endswith('...'),
            answer.endswith(','),
            answer.endswith(';'),
            answer.endswith(' '),
            any(word in answer.lower() for word in ['http', 'www.', '.com']),
            '...' in answer[-10:]
        ]
        
        if any(cut_indicators):
            logger.warning("⚠️  Détection possible de réponse coupée")
            
            last_period = answer.rfind('.')
            last_exclamation = answer.rfind('!')
            last_question = answer.rfind('?')
            
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > 0 and sentence_end >= len(answer) - 5:
                answer = answer[:sentence_end + 1]
            else:
                answer = answer.rstrip(' ,;...')
                if not answer.endswith(('.', '!', '?')):
                    answer += '.'
        
        prevention_phrases = [
            'conseil de prévention',
            'pour prévenir',
            'je recommande',
            'il est important de',
            'n oubliez pas de'
        ]
        
        has_prevention_advice = any(phrase in answer.lower() for phrase in prevention_phrases)
        
        if has_prevention_advice:
            lines = answer.split('. ')
            if len(lines) > 1:
                for i, line in enumerate(lines[1:], 1):
                    if any(phrase in line.lower() for phrase in prevention_phrases):
                        lines[i] = '\n' + lines[i]
                        answer = '. '.join(lines)
                        break
        
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
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
            
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
            
            logger.info(f"✓ Index FAISS créé ({len(self.embeddings)} vecteurs)")
            
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
    """Traite une question et retourne la réponse"""
    
    # Salutations
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
            "method": "salutation",
            "score": None
        }
    
    # Recherche FAISS
    logger.info("🔍 Recherche FAISS...")
    faiss_results = rag_service.search(question)
    
    if not faiss_results:
        return {
            "answer": "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin pour des conseils adaptés. 💗",
            "method": "no_result",
            "score": None
        }
    
    best_result = faiss_results[0]
    similarity = best_result['similarity']
    
    logger.info(f"📊 Meilleure similarité: {similarity:.3f}")
    
    # Décision : Réponse directe vs Génération
    if similarity >= Config.SIMILARITY_THRESHOLD:
        logger.info(f"✅ Haute similarité → Réponse directe")
        answer = best_result['answer']
        
        if len(answer) > Config.MAX_ANSWER_LENGTH:
            answer = answer[:Config.MAX_ANSWER_LENGTH-3] + "..."
        
        return {
            "answer": answer,
            "method": "json_direct",
            "score": float(similarity)
        }
    
    else:
        logger.info(f"🤖 Similarité modérée → Génération Groq")
        
        # Préparer le contexte
        context_parts = []
        for i, result in enumerate(faiss_results[:3], 1):
            answer_truncated = result['answer']
            if len(answer_truncated) > 200:
                answer_truncated = answer_truncated[:197] + "..."
            context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
        
        context = "\n\n".join(context_parts)
        
        # Génération avec Groq
        try:
            if groq_service.available:
                answer = groq_service.generate_response(question, context, history)
                method = "groq_generated"
            else:
                answer = "Je vous recommande de consulter un professionnel de santé pour cette question spécifique. La prévention précoce est essentielle. 💗"
                method = "fallback"
        except Exception as e:
            logger.warning(f"Génération échouée: {str(e)}")
            answer = "Pour des informations précises sur ce sujet, veuillez consulter un médecin ou un centre de santé spécialisé au Bénin. 🌸"
            method = "error_fallback"
        
        return {
            "answer": answer,
            "method": method,
            "score": float(similarity)
        }

# ============================================
# INITIALISATION DES SERVICES (CACHE)
# ============================================

@st.cache_resource
def load_services():
    """Charge les services une seule fois"""
    logger.info("🚀 Chargement des services...")
    groq = GroqService()
    rag = RAGService()
    logger.info("✓ Services chargés")
    return groq, rag

# ============================================
# INTERFACE STREAMLIT
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

groq_service, rag_service = load_services()

# Initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_quick_questions" not in st.session_state:
    st.session_state.show_quick_questions = True
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

# Navbar supérieure
st.markdown("""
<div class="top-navbar">
    <div class="navbar-logo">
        <span>🎀</span> ANONTCHIGAN
    </div>
    <div class="navbar-menu">
        <a href="#">Accueil</a>
        <a href="#">À Propos</a>
        <a href="#" style="background: rgba(255,255,255,0.2);">Chatbot</a>
        <a href="#">Prédiction</a>
        <a href="#">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Header avec avatar
st.markdown("""
<div class="bot-header-card">
    <div class="bot-avatar-circle">🤖</div>
    <div class="bot-header-text">
        <h2>Assistant ANONTCHIGAN</h2>
        <p>En ligne - Prêt à vous aider</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Bannière de bienvenue (affichée uniquement au début)
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-banner">
        <div class="ribbon-icon">🎗️</div>
        <h3>Bienvenue sur ANONTCHIGAN</h3>
        <p>Je suis votre assistant virtuel pour répondre à vos questions sur le cancer du sein.</p>
    </div>
    """, unsafe_allow_html=True)

# Zone de chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Message de bienvenue initial
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="bot-message-wrapper">
        <div class="bot-mini-avatar">🤖</div>
        <div class="bot-message-content">
            Bonjour 👋 Je suis l'assistant ANONTCHIGAN.
            <br><br>
            Je peux vous aider avec des informations sur :
            <ul>
                <li>La prévention du cancer du sein</li>
                <li>Les symptômes à surveiller</li>
                <li>L'auto-examen des seins</li>
                <li>Les ressources disponibles</li>
            </ul>
            <div class="info-box">
                <span class="info-box-icon">⚠️</span>
                <span class="info-box-text"><strong>Important :</strong> Je fournis des informations éducatives. Consultez toujours un médecin pour un diagnostic.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Afficher les messages de la conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        # Formatage du message bot
        content = message["content"]
        
        # Détection des listes à puces et conversion en HTML
        if '•' in content or '\n-' in content:
            lines = content.split('\n')
            formatted_lines = []
            in_list = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('•') or line.startswith('-'):
                    if not in_list:
                        formatted_lines.append('<ul>')
                        in_list = True
                    formatted_lines.append(f'<li>{line[1:].strip()}</li>')
                else:
                    if in_list:
                        formatted_lines.append('</ul>')
                        in_list = False
                    if line:
                        formatted_lines.append(line + '<br>')
            
            if in_list:
                formatted_lines.append('</ul>')
            
            content = ''.join(formatted_lines)
        
        st.markdown(f"""
        <div class="bot-message-wrapper">
            <div class="bot-mini-avatar">🤖</div>
            <div class="bot-message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# Indicateur de frappe (typing indicator)
if st.session_state.is_typing:
    st.markdown("""
    <div class="typing-wrapper">
        <div class="bot-mini-avatar">🤖</div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Questions rapides (affichées uniquement au début)
if st.session_state.show_quick_questions and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="quick-questions-section">
        <div class="quick-questions-title">Questions fréquentes :</div>
    </div>
    """, unsafe_allow_html=True)
    
    quick_questions = [
        "Symptômes du cancer",
        "Auto-examen",
        "Facteurs de risque",
        "Âge de dépistage"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        col_idx = i % 2
        if cols[col_idx].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.show_quick_questions = False
            st.session_state.is_typing = True
            st.rerun()

# Zone d'input fixe en bas
st.markdown('<div class="input-fixed-bottom"><div class="input-wrapper">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Posez-moi une question...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("➤", use_container_width=True, key="send_btn")

st.markdown('</div></div>', unsafe_allow_html=True)

# Traitement de l'envoi
if send_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.show_quick_questions = False
    st.session_state.is_typing = True
    st.rerun()

# Génération de la réponse par le bot
if st.session_state.is_typing and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "user":
        time.sleep(1.5)  # Simule le temps de réflexion
        
        result = process_question(
            st.session_state.messages[-1]["content"],
            [],
            groq_service,
            rag_service
        )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })
        st.session_state.is_typing = False
        st.rerun()

# Script pour scroll automatique vers le bas
st.markdown("""
<script>
    // Scroll automatique
    setTimeout(function() {
        var chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }, 100);
</script>
""", unsafe_allow_html=True)
