import streamlit as st
import json
import logging
from typing import Dict, List, Optional
import random
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uuid

# ============================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN - Prévention Cancer du Sein",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CSS PERSONNALISÉ - DESIGN OCTOBRE ROSE
# ============================================

st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --rose-primary: #E91E63;
        --rose-light: #FCE4EC;
        --rose-dark: #C2185B;
        --violet: #9C27B0;
        --blanc: #FFFFFF;
        --gris-clair: #F5F5F5;
        --gris-fonce: #424242;
    }
    
    /* Reset Streamlit */
    .stApp {
        background: linear-gradient(135deg, #FCE4EC 0%, #E1BEE7 100%);
    }
    
    /* Header personnalisé */
    .main-header {
        background: linear-gradient(135deg, #E91E63 0%, #9C27B0 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Container du chat */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Messages du chat */
    .user-message {
        background: linear-gradient(135deg, #E91E63 0%, #9C27B0 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: #F5F5F5;
        color: #424242;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #E91E63;
        animation: slideInLeft 0.3s ease-out;
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
    
    /* Input personnalisé */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #E91E63;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #9C27B0;
        box-shadow: 0 0 0 3px rgba(233, 30, 99, 0.1);
    }
    
    /* Bouton personnalisé */
    .stButton > button {
        background: linear-gradient(135deg, #E91E63 0%, #9C27B0 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Cards des statistiques */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-top: 4px solid #E91E63;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .stat-card h3 {
        color: #E91E63;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .stat-card p {
        color: #424242;
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #FCE4EC 0%, #F3E5F5 100%);
        border-left: 4px solid #E91E63;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .info-box h4 {
        color: #C2185B;
        margin-top: 0;
        font-size: 1.2rem;
    }
    
    /* Footer */
    .custom-footer {
        background: linear-gradient(135deg, #424242 0%, #616161 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .custom-footer a {
        color: #FCE4EC;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .custom-footer a:hover {
        color: #E91E63;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .user-message, .bot-message {
            max-width: 95%;
        }
        
        .chat-container {
            padding: 1rem;
        }
    }
    
    /* Cacher les éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION ET LOGGING (CODE ORIGINAL)
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ANONTCHIGAN")

class Config:
    """Configuration optimisée pour éviter les coupures"""
    SIMILARITY_THRESHOLD = 0.75
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600
    FAISS_RESULTS_COUNT = 3
    MIN_ANSWER_LENGTH = 30

# ============================================
# SERVICE GROQ (CODE ORIGINAL)
# ============================================

class GroqService:
    
    def __init__(self):
        self.client = None
        self.available = False
        self._initialize_groq()
    
    def _initialize_groq(self):
        try:
            from groq import Groq
            
            api_key = "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"
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
            # Préparer le contexte optimisé
            context_short = self._prepare_context(context)
            
            # Préparer les messages
            messages = self._prepare_messages(question, context_short, history)
            
            logger.info("🤖 Génération avec Groq...")
            
            # AUGMENTER SIGNIFICATIVEMENT les tokens pour éviter les coupures
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                top_p=0.9,
            )
            
            answer = response.choices[0].message.content.strip()
            answer = self._clean_response(answer)
            
            # Validation renforcée
            if not self._is_valid_answer(answer):
                raise ValueError("Réponse trop courte")
                
            # Vérification et correction des coupures
            answer = self._ensure_complete_response(answer)
            
            logger.info(f"✓ Réponse générée ({len(answer)} caractères)")
            return answer
            
        except Exception as e:
            logger.error(f"Erreur Groq: {str(e)}")
            raise
    
    def _prepare_context(self, context: str) -> str:
        """Prépare un contexte concis"""
        lines = context.split('\n')[:5]
        context_short = '\n'.join(lines)
        if len(context_short) > Config.MAX_CONTEXT_LENGTH:
            context_short = context_short[:Config.MAX_CONTEXT_LENGTH-3] + "..."
        return context_short
    
    def _prepare_messages(self, question: str, context: str, history: List[Dict]) -> List[Dict]:
        """Prépare les messages avec instructions claires contre les coupures"""
        
        system_prompt = f"""Tu es ANONTCHIGAN, assistante IA professionnelle spécialisée dans la sensibilisation au cancer du sein au Bénin.

CONTEXTE À UTILISER :
{context}

RÈGLES CRITIQUES :
1. FOURNIR DES RÉPONSES COMPLÈTES - ne JAMAIS couper une phrase en milieu de mot
2. Si tu donnes un conseil de prévention, aller deux fois à la ligne AVANT de le donner
3. Tes réponses doivent se terminer naturellement par un point final
4. Tes créateurs sont Judicaël Karol DOBOEVI, Ursus Hornel GBAGUIDI, Abel Kokou KPOCOUTA et Josaphat ADJELE, touts membres
du club d'IA de l'ENSGMM (Ecole Nationale Supérieure de Génie Mathématique et Modélisation) d'Abomey. 

STYLE :
- Professionnel, clair, empathique
- Réponses directes sans formules introductives
- CONCIS mais COMPLET
- Humour humain léger 1 fois sur 3 réponses
- Emojis à utiliser fréquemment : 💗 🌸 😊 🇧🇯

STRUCTURE DES RÉPONSES :
1. Réponse principale basée sur le contexte
2- N'invente PAS d'informations
3- Si le contexte ne couvre pas la question, recommande de consulter un professionnel de santé au Bénin
4- ENSGMM se définit comme l'Ecole Nationale Supérieure de Génie Mathématique et Modélisation 
5- Utilise des connecteurs logiques pour fluidifier et produire des réponses naturelles et cohérentes

DIRECTIVES ANTI-CONPUTURE :
- Vérifie que ta réponse est complète avant de terminer
- Ne coupe PAS en milieu de phrase ou de mot
- Utilise "Atassa!" ou "Atassaaaaa!" en debut de phrase en cas dh'humour et d'étonnemnt extrême
- Termine par un point final approprié
- Si tu mentionnes des noms (créateurs, etc.), assure-toi qu'ils sont COMPLETS

Conseils de prévention : seulement si pertinents et si demandés."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Historique récent
        for msg in history[-4:]:
            messages.append(msg)
        
        # Question actuelle avec instruction explicite
        messages.append({
            "role": "user", 
            "content": f"QUESTION: {question}\n\nIMPORTANT : Réponds de façon COMPLÈTE sans couper ta réponse. Termine par un point final. Si conseil de prévention, va à la ligne avant."
        })
        
        return messages
    
    def _clean_response(self, answer: str) -> str:
        """Nettoie la réponse en gardant la personnalité"""
        
        # Supprimer les introductions verbeuses
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
        """Valide que la réponse est acceptable"""
        return (len(answer) >= Config.MIN_ANSWER_LENGTH and 
                not answer.lower().startswith(('je ne sais pas', 'désolé', 'sorry')))
    
    def _ensure_complete_response(self, answer: str) -> str:
        """Garantit que la réponse est complète et non coupée"""
        if not answer:
            return answer
            
        # Détecter les signes de coupure
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
            
            # Trouver la dernière phrase complète
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
        
        # Formater les conseils de prévention avec saut de ligne
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
# SERVICES RAG (CODE ORIGINAL)
# ============================================

class RAGService:
    """Service RAG avec recherche améliorée"""
    
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
        """Recherche optimisée dans FAISS"""
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

class ConversationManager:
    """Gestionnaire de conversations"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
    
    def get_history(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])
    
    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({"role": role, "content": content})
        
        if len(self.conversations[user_id]) > Config.MAX_HISTORY_LENGTH * 2:
            self.conversations[user_id] = self.conversations[user_id][-Config.MAX_HISTORY_LENGTH * 2:]

# ============================================
# INITIALISATION DES SERVICES
# ============================================

@st.cache_resource
def initialize_services():
    """Initialise les services une seule fois"""
    groq = GroqService()
    rag = RAGService()
    conv = ConversationManager()
    return groq, rag, conv

groq_service, rag_service, conversation_manager = initialize_services()

# ============================================
# INITIALISATION SESSION STATE
# ============================================

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================
# FONCTION DE TRAITEMENT (CODE ORIGINAL)
# ============================================

def process_question(question: str, user_id: str) -> Dict:
    """Traite la question et retourne la réponse"""
    try:
        history = conversation_manager.get_history(user_id)
        
        # Gestion des salutations
        salutations = ["cc","bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
        question_lower = question.lower().strip()
        
        if any(salut == question_lower for salut in salutations):
            responses = [
                "Je suis ANONTCHIGAN, assistante pour la sensibilisation au cancer du sein. Comment puis-je vous aider ? 💗",
                "Bonjour ! Je suis ANONTCHIGAN. Que souhaitez-vous savoir sur le cancer du sein ? 🌸",
                "ANONTCHIGAN à votre service. Posez-moi vos questions sur la prévention du cancer du sein. 😊"
            ]
            answer = random.choice(responses)
            
            conversation_manager.add_message(user_id, "user", question)
            conversation_manager.add_message(user_id, "assistant", answer)
            
            return {
                "answer": answer,
                "status": "success",
                "method": "salutation"
            }
        
        # Recherche FAISS
        logger.info("🔍 Recherche FAISS...")
        faiss_results = rag_service.search(question)
        
        if not faiss_results:
            answer = "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin pour des conseils adaptés. 💗"
            conversation_manager.add_message(user_id, "user", question)
            conversation_manager.add_message(user_id, "assistant", answer)
            
            return {
                "answer": answer,
                "status": "info",
                "method": "no_result"
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
            
            conversation_manager.add_message(user_id, "user", question)
            conversation_manager.add_message(user_id, "assistant", answer)
            
            return {
                "answer": answer,
                "status": "success",
                "method": "json_direct",
                "score": float(similarity),
                "matched_question": best_result['question']
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
                    generated_answer = groq_service.generate_response(question, context, history)
                else:
                    generated_answer = "Je vous recommande de consulter un professionnel de santé pour cette question spécifique. La prévention précoce est essentielle. 💗"
            except Exception as e:
                logger.warning(f"Génération échouée: {str(e)}")
                generated_answer = "Pour des informations précises sur ce sujet, veuillez consulter un médecin ou un centre de santé spécialisé au Bénin. 🌸"
            
            conversation_manager.add_message(user_id, "user", question)
            conversation_manager.add_message(user_id, "assistant", generated_answer)
            
            return {
                "answer": generated_answer,
                "status": "success",
                "method": "groq_generated",
                "score": float(similarity),
                "context_used": len(faiss_results[:3])
            }
            
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
        error_message = "Désolé, une erreur s'est produite. Veuillez réessayer."
        
        conversation_manager.add_message(user_id, "user", question)
        conversation_manager.add_message(user_id, "assistant", error_message)
        
        return {
            "answer": error_message,
            "status": "error",
            "method": "error"
        }

# ============================================
# INTERFACE STREAMLIT
# ============================================

def display_message(message: dict, is_user: bool):
    """Affiche un message dans le chat"""
    if is_user:
        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🎀 {message["content"]}</div>', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎀 ANONTCHIGAN</h1>
    <p>Votre Assistante IA pour la Sensibilisation au Cancer du Sein au Bénin</p>
</div>
""", unsafe_allow_html=True)

# Tabs pour navigation
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📊 Informations", "ℹ️ À Propos"])

# ============================================
# TAB 1: CHATBOT
# ============================================

with tab1:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Zone de messages
    chat_placeholder = st.container()
    
    with chat_placeholder:
        if len(st.session_state.chat_history) == 0:
            st.markdown("""
            <div class="info-box">
                <h4>👋 Bienvenue !</h4>
                <p>Je suis ANONTCHIGAN, votre assistante dédiée à la prévention du cancer du sein. 
                Posez-moi vos questions sur les symptômes, la prévention, le dépistage ou tout autre sujet lié au cancer du sein. 💗</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                display_message(msg, msg["role"] == "user")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone de saisie
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Votre question...",
            key="user_input",
            placeholder="Posez votre question sur le cancer du sein...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Envoyer 💗")
    
    # Traitement de l'envoi
    if send_button and user_input.strip():
        # Ajouter le message utilisateur
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Afficher un loader
        with st.spinner("ANONTCHIGAN réfléchit..."):
            # Obtenir la réponse
            response = process_question(user_input, st.session_state.user_id)
            
            # Ajouter la réponse du bot
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"]
            })
        
        # Rerun pour afficher les nouveaux messages
        st.rerun()
    
    # Bouton pour effacer l'historique
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()

# ============================================
# TAB 2: INFORMATIONS
# ============================================

with tab2:
    st.markdown("### 📊 Statistiques Clés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>1/8</h3>
            <p>Femmes développent un cancer du sein</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>90%</h3>
            <p>Taux de guérison si détecté tôt</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>40+</h3>
            <p>Âge recommandé pour le dépistage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🌸 Conseils de Prévention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🏃‍♀️ Activité Physique</h4>
            <p>Pratiquez au moins 30 minutes d'exercice modéré par jour pour réduire les risques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🥗 Alimentation Saine</h4>
            <p>Privilégiez les fruits, légumes et limitez l'alcool et les aliments transformés.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Auto-Examen</h4>
            <p>Examinez vos seins mensuellement pour détecter toute anomalie.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>👩‍⚕️ Dépistage Régulier</h4>
            <p>Consultez un professionnel pour une mammographie à partir de 40 ans.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TAB 3: À PROPOS
# ============================================

with tab3:
    st.markdown("### 🎀 À Propos d'ANONTCHIGAN")
    
    st.markdown("""
    <div class="info-box">
        <h4>Notre Mission</h4>
        <p>ANONTCHIGAN est une assistante IA développée pour sensibiliser et informer sur le cancer du sein au Bénin. 
        Notre objectif est de rendre l'information médicale accessible à tous et de promouvoir le dépistage précoce.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 👨‍💻 Équipe de Développement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Créateurs :**
        - Judicaël Karol DOBOEVI
        - Ursus Hornel GBAGUIDI
        - Abel Kokou KPOCOUTA
        - Josaphat ADJELE
        """)
    
    with col2:
        st.markdown("""
        **Institution :**
        - ENSGMM (École Nationale Supérieure de Génie Mathématique et Modélisation)
        - Abomey, Bénin 🇧🇯
        - Club d'Intelligence Artificielle
        """)
    
    st.markdown("""
    <div class="info-box">
        <h4>⚙️ Technologies Utilisées</h4>
        <p>
        • Intelligence Artificielle (LLM Groq)<br>
        • RAG (Retrieval-Augmented Generation)<br>
        • FAISS pour la recherche sémantique<br>
        • Sentence Transformers pour les embeddings<br>
        • Streamlit pour l'interface
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ État du Système")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Disponible" if groq_service.available else "❌ Indisponible"
        st.metric("Service Groq", status)
    
    with col2:
        st.metric("Base de données", f"{len(rag_service.questions_data)} questions")
    
    with col3:
        st.metric("Conversations actives", len(conversation_manager.conversations))

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="custom-footer">
    <p><strong>ANONTCHIGAN</strong> - Octobre Rose 2024 🎀</p>
    <p>Développé avec 💗 par le Club IA de l'ENSGMM</p>
    <p><small>Pour toute urgence médicale, consultez immédiatement un professionnel de santé</small></p>
</div>
""", unsafe_allow_html=True)

# ============================================
# INFORMATIONS DE DÉMARRAGE
# ============================================

logger.info("\n" + "="*50)
logger.info("✓ ANONTCHIGAN STREAMLIT - Prêt!")
logger.info(f"  - Génération: {'Groq ⚡' if groq_service.available else 'Fallback'}")
logger.info(f"  - Questions chargées: {len(rag_service.questions_data)}")
logger.info(f"  - User ID: {st.session_state.user_id}")
logger.info("="*50 + "\n")
