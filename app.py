import json
import os
import logging
from typing import Dict, List, Optional
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from streamlit.web import cli as stcli
import sys

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN API",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

groq_service, rag_service = load_services()

# ============================================
# GESTION DES PARAMÈTRES URL
# ============================================

# Vérifier si c'est un appel API via les query params
query_params = st.query_params

if "api" in query_params and query_params["api"] == "true":
    # MODE API - Pas d'interface, juste réponse JSON
    if "question" in query_params:
        question = query_params["question"]
        user_id = query_params.get("user_id", f"user_{random.randint(1000, 9999)}")
        
        try:
            result = process_question(question, [], groq_service, rag_service)
            
            response_data = {
                "success": True,
                "answer": result["answer"],
                "method": result["method"],
                "similarity_score": result["score"],
                "user_id": user_id
            }
            
            st.json(response_data)
            st.stop()
            
        except Exception as e:
            error_data = {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
            st.json(error_data)
            st.stop()
    else:
        st.json({
            "success": False,
            "error": "Paramètre 'question' manquant"
        })
        st.stop()

# ============================================
# INTERFACE STREAMLIT NORMALE
# ============================================

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #ff6b9d 0%, #c44569 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .api-info {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin-bottom: 1rem;
    }
    .api-code {
        background: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.85em;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💗 ANONTCHIGAN API</h1>
    <p>Assistante IA pour la sensibilisation au cancer du sein au Bénin 🇧🇯</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Informations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{len(rag_service.questions_data)}</h3>
            <p>Questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        groq_status = "✅ Activé" if groq_service.available else "❌ Désactivé"
        st.markdown(f"""
        <div class="stat-box">
            <h3>{groq_status}</h3>
            <p>Groq AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Documentation API
    st.markdown("### 🔗 Utiliser l'API")
    
    # Récupérer l'URL de l'app
    try:
        app_url = st.secrets.get("app_url", "https://votre-app.streamlit.app")
    except:
        app_url = "https://votre-app.streamlit.app"
    
    st.markdown(f"""
    <div class="api-info">
        <h4>Méthode GET</h4>
        <p>Envoyez vos questions via URL :</p>
        <div class="api-code">
{app_url}/?api=true&question=Votre+question
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="api-info">
        <h4>📝 Exemple JavaScript</h4>
        <div class="api-code">
const question = "Symptômes cancer sein";<br>
const url = `{URL}/?api=true&question=${encodeURIComponent(question)}`;<br>
<br>
fetch(url)<br>
&nbsp;&nbsp;.then(res => res.json())<br>
&nbsp;&nbsp;.then(data => console.log(data.answer));
        </div>
    </div>
    """.replace("{URL}", app_url), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="api-info">
        <h4>🐍 Exemple Python</h4>
        <div class="api-code">
import requests<br>
import urllib.parse<br>
<br>
question = "Symptômes cancer sein"<br>
url = f"{URL}/?api=true&question={urllib.parse.quote(question)}"<br>
<br>
response = requests.get(url)<br>
data = response.json()<br>
print(data['answer'])
        </div>
    </div>
    """.replace("{URL}", app_url), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👥 Créateurs
    - Judicaël Karol DOBOEVI
    - Ursus Hornel GBAGUIDI
    - Abel Kokou KPOCOUTA
    - Josaphat ADJELE
    
    **Club d'IA - ENSGMM Abomey**
    """)
    
    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.messages = []
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"
        st.session_state.conversation_history = []
        st.rerun()

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{random.randint(1000, 9999)}"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if question := st.chat_input("Posez votre question sur le cancer du sein..."):
    # Ajouter la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Traiter la question
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            try:
                result = process_question(
                    question, 
                    st.session_state.conversation_history,
                    groq_service,
                    rag_service
                )
                
                answer = result["answer"]
                method = result["method"]
                score = result["score"]
                
                # Afficher la réponse
                st.markdown(answer)
                
                # Afficher les métadonnées (optionnel)
                with st.expander("ℹ️ Détails de la réponse"):
                    st.write(f"**Méthode:** {method}")
                    if score is not None:
                        st.write(f"**Score de similarité:** {score:.3f}")
                    st.write(f"**User ID:** {st.session_state.user_id}")
                
                # Ajouter à l'historique
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Mettre à jour l'historique de conversation
                st.session_state.conversation_history.append({"role": "user", "content": question})
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
                # Limiter l'historique
                if len(st.session_state.conversation_history) > Config.MAX_HISTORY_LENGTH * 2:
                    st.session_state.conversation_history = st.session_state.conversation_history[-Config.MAX_HISTORY_LENGTH * 2:]
                
            except Exception as e:
                error_message = f"❌ Erreur: {str(e)}"
                st.error(error_message)
                logger.error(error_message)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>ANONTCHIGAN v2.3.0 - Développé avec ❤️ par le Club d'IA de l'ENSGMM</p>
    <p>Pour la sensibilisation au cancer du sein au Bénin 🇧🇯</p>
</div>
""", unsafe_allow_html=True)
