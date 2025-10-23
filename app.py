import json
import os
import logging
from typing import Dict, List, Optional
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import streamlit.components.v1 as components

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ANONTCHIGAN",
    page_icon="💗🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
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
# GESTION DES PARAMÈTRES URL (MODE API)
# ============================================

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
# INTERFACE HTML PERSONNALISÉE
# ============================================

html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --rose-primary: #E91E63;
    --rose-dark: #C2185B;
    --violet: #9C27B0;
    --blanc: #FFFFFF;
    --gris: #424242;
}
body {
    font-family: 'Segoe UI', sans-serif;
    background: #fff;
    color: var(--gris);
}
nav {
    background: linear-gradient(135deg, var(--rose-primary), var(--violet));
    padding: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    position: sticky;
    top: 0;
    z-index: 1000;
}
.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
}
.logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #fff;
    font-size: 1.5rem;
    font-weight: bold;
    text-decoration: none;
}
.logo i { font-size: 2rem; }
.nav-menu {
    display: flex;
    gap: 2rem;
    list-style: none;
}
.nav-menu a {
    color: #fff;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    transition: all 0.3s;
    font-weight: 500;
}
.nav-menu a:hover, .nav-menu a.active {
    background: rgba(255,255,255,0.2);
    transform: translateY(-2px);
}
.menu-toggle {
    display: none;
    background: none;
    border: none;
    color: #fff;
    font-size: 1.5rem;
    cursor: pointer;
}
.chat-container {
    max-width: 1000px;
    margin: 1rem auto;
    height: calc(100vh - 180px);
    min-height: 500px;
    display: flex;
    flex-direction: column;
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    overflow: hidden;
}
.chat-header {
    background: linear-gradient(135deg, var(--rose-primary), var(--violet));
    color: #fff;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.chat-header-avatar {
    width: 50px;
    height: 50px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    background: linear-gradient(to bottom, #f8f9fa, #fff);
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.message {
    display: flex;
    gap: 1rem;
    animation: slideIn 0.3s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.message.bot .message-avatar {
    background: linear-gradient(135deg, var(--rose-primary), var(--violet));
    color: #fff;
}
.message.user {
    flex-direction: row-reverse;
}
.message.user .message-avatar {
    background: #f5f5f5;
    color: var(--gris);
}
.message-content {
    max-width: 75%;
    padding: 1rem 1.5rem;
    border-radius: 20px;
    line-height: 1.6;
}
.message.bot .message-content {
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-bottom-left-radius: 5px;
}
.message.user .message-content {
    background: linear-gradient(135deg, var(--rose-primary), var(--violet));
    color: #fff;
    border-bottom-right-radius: 5px;
}
.message-time {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 0.3rem;
}
.typing-indicator {
    display: none;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
}
.typing-indicator.active { display: flex; }
.typing-dots {
    display: flex;
    gap: 0.3rem;
    padding: 1rem 1.5rem;
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.typing-dots span {
    width: 8px;
    height: 8px;
    background: var(--rose-primary);
    border-radius: 50%;
    animation: typing 1.4s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
    30% { transform: translateY(-10px); opacity: 1; }
}
.chat-input-container {
    padding: 1.5rem;
    background: #fff;
    border-top: 1px solid #e0e0e0;
}
.chat-input-wrapper {
    display: flex;
    gap: 1rem;
    align-items: flex-end;
}
.chat-input {
    flex: 1;
    padding: 1rem 1.5rem;
    border: 2px solid #e0e0e0;
    border-radius: 25px;
    font-size: 1rem;
    font-family: inherit;
    resize: none;
    max-height: 120px;
    min-height: 50px;
}
.chat-input:focus {
    outline: none;
    border-color: var(--rose-primary);
    box-shadow: 0 0 0 3px rgba(233,30,99,0.1);
}
.send-button {
    width: 50px;
    height: 50px;
    border: none;
    background: linear-gradient(135deg, var(--rose-primary), var(--violet));
    color: #fff;
    border-radius: 50%;
    font-size: 1.2rem;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    justify-content: center;
}
.send-button:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px rgba(233,30,99,0.4);
}
.send-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
.quick-questions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}
.quick-question-btn {
    padding: 0.5rem 1rem;
    background: #fff;
    border: 2px solid var(--rose-primary);
    color: var(--rose-primary);
    border-radius: 20px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.3s;
}
.quick-question-btn:hover {
    background: var(--rose-primary);
    color: #fff;
    transform: translateY(-2px);
}
.welcome-message {
    text-align: center;
    padding: 2rem;
}
.welcome-message i {
    font-size: 4rem;
    color: var(--rose-primary);
    margin-bottom: 1rem;
}
.disclaimer {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #f57c00;
}
.source-badge {
    font-size: 0.75rem;
    opacity: 0.8;
    margin-top: 0.5rem;
    padding: 0.3rem 0.6rem;
    background: #f5f5f5;
    border-radius: 12px;
    display: inline-block;
}
.source-badge.json_direct { background: #e8f5e8; color: #2e7d32; }
.source-badge.groq_generated { background: #e3f2fd; color: #1565c0; }
.source-badge.salutation { background: #f3e5f5; color: #7b1fa2; }
@media (max-width: 768px) {
    .nav-menu {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, var(--rose-primary), var(--violet));
        flex-direction: column;
        padding: 0.5rem 0;
        gap: 0;
        display: none;
        z-index: 999;
    }
    .nav-menu.active { display: flex; }
    .nav-menu a { padding: 0.8rem 1.5rem; width: 100%; }
    .menu-toggle { display: block; }
    .chat-container { margin: 0; height: calc(100vh - 120px); border-radius: 0; }
    .message-content { max-width: 85%; }
}
</style>
</head>
<body>

<nav>
<div class="nav-container">
<a href="https://abel123.pythonanywhere.com/" class="logo" target="_blank">
<i class="fas fa-ribbon"></i>
<span>ANONTCHIGAN</span>
</a>
<button class="menu-toggle" onclick="toggleMenu()">
<i class="fas fa-bars"></i>
</button>
<ul class="nav-menu" id="navMenu">
<li><a href="https://abel123.pythonanywhere.com/contact/" target="_blank">Contact</a></li>
</ul>
</div>
</nav>

<div class="chat-container">
<div class="chat-header">
<div class="chat-header-avatar"><i class="fas fa-robot"></i></div>
<div class="chat-header-info">
<h2>Assistant ANONTCHIGAN</h2>
<p><i class="fas fa-circle" style="color: #4caf50; font-size: 0.6rem;"></i> En ligne</p>
</div>
</div>

<div class="chat-messages" id="chatMessages">
<div class="welcome-message">
<i class="fas fa-ribbon"></i>
<h3 style="color: var(--rose-dark); margin-bottom: 0.5rem;">Bienvenue sur ANONTCHIGAN</h3>
<p>Je suis votre assistant virtuel pour le cancer du sein.</p>
</div>

<div class="message bot">
<div class="message-avatar"><i class="fas fa-robot"></i></div>
<div>
<div class="message-content">
<p>Bonjour ! 👋 Je suis l'assistant ANONTCHIGAN.</p>
<p style="margin-top: 0.5rem;">Je peux vous aider avec :</p>
<ul style="margin: 0.5rem 0 0.5rem 1.5rem;">
<li>La prévention du cancer du sein</li>
<li>Les symptômes à surveiller</li>
<li>L'auto-examen des seins</li>
<li>Les ressources disponibles</li>
</ul>
<div class="disclaimer" style="margin-top: 1rem;">
<i class="fas fa-info-circle"></i>
<strong>Important :</strong> Consultez toujours un médecin pour un diagnostic.
</div>
</div>
<div class="message-time">Maintenant</div>
</div>
</div>

<div style="padding: 0 1rem;">
<p style="font-size: 0.9rem; margin-bottom: 0.5rem;">Questions fréquentes :</p>
<div class="quick-questions">
<button class="quick-question-btn" onclick="sendQuickQuestion('Quels sont les symptômes du cancer du sein ?')">Symptômes</button>
<button class="quick-question-btn" onclick="sendQuickQuestion('Comment faire l\\'auto-examen des seins ?')">Auto-examen</button>
<button class="quick-question-btn" onclick="sendQuickQuestion('Quels sont les facteurs de risque ?')">Facteurs de risque</button>
<button class="quick-question-btn" onclick="sendQuickQuestion('À partir de quel âge faire un dépistage ?')">Âge de dépistage</button>
</div>
</div>

<div class="typing-indicator" id="typingIndicator">
<div class="message-avatar" style="background: linear-gradient(135deg, var(--rose-primary), var(--violet)); color: #fff;">
<i class="fas fa-robot"></i>
</div>
<div class="typing-dots">
<span></span>
<span></span>
<span></span>
</div>
</div>
</div>

<div class="chat-input-container">
<div class="chat-input-wrapper">
<textarea id="chatInput" class="chat-input" placeholder="Posez votre question ici..." rows="1"></textarea>
<button class="send-button" id="sendButton" onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
</div>
</div>
</div>

<script>
// Variables globales
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');

// Fonction pour envoyer un message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.disabled = true;
    sendButton.disabled = true;
    typingIndicator.classList.add('active');
    scrollToBottom();

    try {
        const apiUrl = window.location.origin + window.location.pathname + '?api=true&question=' + encodeURIComponent(message);
        
        console.log('🔍 Envoi à:', apiUrl);
        
        const response = await fetch(apiUrl);
        const data = await response.json();

        console.log('✅ Réponse:', data);

        typingIndicator.classList.remove('active');

        if (data.success && data.answer) {
            const sourceText = getSourceText(data.method);
            addMessage(data.answer, 'bot', sourceText, data.method);
        } else {
            addMessage("❌ Erreur: " + (data.error || 'Réponse invalide'), 'bot');
        }

    } catch (error) {
        typingIndicator.classList.remove('active');
        addMessage("❌ Erreur de connexion: " + error.message, 'bot');
        console.error('❌ Erreur:', error);
    }

    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
}

// Fonction pour envoyer une question rapide
function sendQuickQuestion(question) {
    if (chatInput) {
        chatInput.value = question;
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        sendMessage();
    }
}

// Fonction pour ajouter un message
function addMessage(text, sender, source, sourceType) {
    if (!chatMessages || !typingIndicator) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + sender;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

    const contentWrapper = document.createElement('div');
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = formatMessage(text);

    if (source && sender === 'bot') {
        const sourceTag = document.createElement('div');
        sourceTag.className = 'source-badge ' + sourceType;
        sourceTag.innerHTML = '<i class="fas fa-info-circle"></i> ' + source;
        content.appendChild(sourceTag);
    }

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

    contentWrapper.appendChild(content);
    contentWrapper.appendChild(time);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentWrapper);

    chatMessages.insertBefore(messageDiv, typingIndicator);
    scrollToBottom();
}

// Fonction pour formater les messages
function formatMessage(text) {
    if (!text) return '';
    
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<strong>$1</strong>');
    text = text.replace(/\n/g, '<br>');
    text = text.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
    text = text.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    
    if (text.includes('<li>') && !text.includes('<ul>')) {
        text = text.replace(/(<li>.*?<\/li>)/gs, '<ul style="margin: 0.5rem 0 0.5rem 1.5rem;">$1</ul>');
    }
    
    text = text.replace(/💗/g, '<span style="color: #E91E63;">💗</span>');
    text = text.replace(/👋/g, '<span style="font-size: 1.2em;">👋</span>');
    text = text.replace(/🌸/g, '<span style="font-size: 1.1em;">🌸</span>');
    
    return text;
}

// Fonction pour obtenir le texte de la source
function getSourceText(method) {
    const map = {
        'salutation': '🤝 Accueil',
        'json_direct': '📚 Base FAQ',
        'groq_generated': '🤖 IA Groq',
        'no_result': 'ℹ Info',
        'fallback': '💡 Conseil',
        'error_fallback': '⚠ Erreur'
    };
    return map[method] || '💗 ANONTCHIGAN';
}

// Fonction pour scroller vers le bas
function scrollToBottom() {
    if (chatMessages) {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }
}

// Fonction pour toggle le menu
function toggleMenu() {
    const menu = document.getElementById('navMenu');
    if (menu) menu.classList.toggle('active');
}

// Event listeners
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            const menu = document.getElementById('navMenu');
            if (menu) menu.classList.remove('active');
        }
    });
});

if (chatInput) {
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    
    chatInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
}

window.addEventListener('load', () => {
    if (chatInput) {
        chatInput.focus();
        console.log('✅ Interface chargée et prête');
    }
});
</script>
</body>
</html>
"""

components.html(html_content, height=800, scrolling=False)

# ============================================
# INITIALISATION SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{random.randint(1000, 9999)}"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
<ul class="nav-menu" id="navMenu">
<li><a href="https://abel123.pythonanywhere.com/" target="_blank">Accueil</a></li>
<li><a href="https://abel123.pythonanywhere.com/a-propos/" target="_blank">À Propos</a></li>
<li><a href="#" class="active">Chatbot</a></li>
<li><a href="https://abel123.pythonanywhere.com/predictor/" target="_blank">Prédiction</a></li>
<li><a href="https://abel123.pythonanywhere.com/contact/" target="_blank">Contact</a></li>
</ul>
