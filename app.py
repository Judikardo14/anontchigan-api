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

# ============================================
# DÉTECTION MODE API
# ============================================

# Désactiver tous les éléments visuels Streamlit
st.set_page_config(
    page_title="ANONTCHIGAN API",
    page_icon="💗",
    layout="centered"
)

# Récupérer les paramètres URL
query_params = st.query_params

# MODE API : Renvoie uniquement du JSON
if "question" in query_params and query_params.get("format") == "json":
    # Masquer complètement l'interface Streamlit
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {background-color: white;}
        .block-container {padding: 0 !important; max-width: 100% !important;}
    </style>
    """, unsafe_allow_html=True)
    
    question = query_params.get("question")
    user_id = query_params.get("user_id", f"user_{random.randint(1000, 9999)}")
    
    try:
        # Charger les services
        groq_service, rag_service = load_services()
        
        # Traiter la question
        result = process_question(question, [], groq_service, rag_service)
        
        response_data = {
            "success": True,
            "answer": result["answer"],
            "method": result["method"],
            "similarity_score": result["score"],
            "user_id": user_id,
            "question": question
        }
        
    except Exception as e:
        response_data = {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "question": question
        }
    
    # Afficher UNIQUEMENT le JSON
    st.code(json.dumps(response_data, ensure_ascii=False, indent=2), language="json")
    
    # Bouton pour copier
    st.download_button(
        label="📋 Télécharger la réponse JSON",
        data=json.dumps(response_data, ensure_ascii=False, indent=2),
        file_name="response.json",
        mime="application/json"
    )
    
    st.stop()

# ============================================
# INTERFACE STREAMLIT NORMALE (si pas en mode API)
# ============================================

groq_service, rag_service = load_services()

st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <h1>💗 ANONTCHIGAN API</h1>
    <p>Assistante IA pour la sensibilisation au cancer du sein au Bénin 🇧🇯</p>
</div>
""", unsafe_allow_html=True)

st.info("""
### 🔗 Utiliser l'API en mode JSON

Pour obtenir une réponse en JSON pur, ajoutez `?format=json&question=VotreQuestion` à l'URL.

**Exemple:**
```
https://votre-app.streamlit.app/?format=json&question=Symptômes+cancer+sein
```
""")

# Interface de test
st.markdown("### 🧪 Tester l'API")

test_question = st.text_input("Entrez une question pour tester:", placeholder="Ex: Quels sont les symptômes du cancer du sein ?")

if st.button("🚀 Tester"):
    if test_question:
        with st.spinner("Traitement..."):
            try:
                result = process_question(test_question, [], groq_service, rag_service)
                
                response_data = {
                    "success": True,
                    "answer": result["answer"],
                    "method": result["method"],
                    "similarity_score": result["score"],
                    "question": test_question
                }
                
                st.success("✅ Réponse générée !")
                st.json(response_data)
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer une question")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>ANONTCHIGAN v3.0.0 - Mode API JSON</p>
    <p>Développé par le Club d'IA de l'ENSGMM 🇧🇯</p>
</div>
""", unsafe_allow_html=True)
