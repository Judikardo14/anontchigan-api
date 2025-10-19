import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser FastAPI
app = FastAPI(
    title="ANONTCHIGAN API",
    description="Assistante IA béninoise pour la sensibilisation au cancer du sein",
    version="1.0.0"
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle pour les requêtes
class Query(BaseModel):
    question: str

# Variables globales (initialisées au startup)
embedding_model = None
index = None
questions_data = []
USE_GROQ = False
groq_client = None

# ============================================
# INITIALISATION AU DÉMARRAGE
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialise tous les composants au démarrage"""
    global embedding_model, index, questions_data, USE_GROQ, groq_client
    
    logger.info("="*50)
    logger.info("🚀 Démarrage d'ANONTCHIGAN...")
    logger.info("="*50)
    
    try:
        # 1. Configuration Groq
        logger.info("🔧 Configuration de Groq...")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if GROQ_API_KEY:
            try:
                from groq import Groq
                groq_client = Groq(api_key=GROQ_API_KEY)
                
                # Test rapide
                test_response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model="llama-3.1-8b-instant",
                    max_tokens=5,
                )
                USE_GROQ = True
                logger.info("✓ Groq configuré")
            except Exception as e:
                logger.warning(f"⚠️ Groq non disponible: {str(e)[:50]}")
                USE_GROQ = False
        else:
            logger.warning("⚠️ GROQ_API_KEY non définie")
        
        # 2. Chargement des données
        logger.info("📚 Chargement de cancer_sein.json...")
        try:
            with open('cancer_sein.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error("❌ cancer_sein.json non trouvé")
            raise
        
        all_texts = []
        for item in data:
            questions_data.append({
                'question_originale': item['question'],
                'question_normalisee': item['question'].lower().strip(),
                'answer': item['answer']
            })
            all_texts.append(f"Question: {item['question']}\nRéponse: {item['answer']}")
        
        logger.info(f"✓ {len(questions_data)} questions chargées")
        
        # 3. Initialisation des embeddings (MODÈLE LÉGER)
        logger.info("🔍 Chargement du modèle d'embeddings...")
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        logger.info("✓ Modèle chargé")
        
        # 4. Création de l'index FAISS
        logger.info("📊 Création de l'index FAISS...")
        embeddings = embedding_model.encode(all_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"✓ Index FAISS créé ({len(embeddings)} vecteurs)")
        
        # Nettoyer
        del embeddings
        del all_texts
        import gc
        gc.collect()
        
        logger.info("="*50)
        logger.info("✓ ANONTCHIGAN PRÊT !")
        logger.info(f"  - Groq: {'Activé ⚡' if USE_GROQ else 'Désactivé'}")
        logger.info(f"  - Questions: {len(questions_data)}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}", exc_info=True)
        raise

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "message": "Bienvenue sur ANONTCHIGAN API 💗",
        "status": "running",
        "groq_enabled": USE_GROQ,
        "questions_loaded": len(questions_data)
    }

@app.get("/health")
async def health_check():
    """Health check"""
    if embedding_model is None or index is None:
        raise HTTPException(status_code=503, detail="Service initializing...")
    
    return {
        "status": "healthy",
        "groq_enabled": USE_GROQ,
        "questions_loaded": len(questions_data)
    }

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def search_faiss(query: str, k: int = 3):
    """Recherche dans FAISS"""
    if embedding_model is None or index is None:
        return []
    
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(questions_data):
            similarity = 1 / (1 + distances[0][i])
            results.append({
                'question': questions_data[idx]['question_originale'],
                'answer': questions_data[idx]['answer'],
                'similarity': similarity,
                'distance': distances[0][i]
            })
    
    return results

def find_by_keywords(user_question_lower):
    """Recherche par mots-clés"""
    keyword_patterns = {
        'identite': {
            'keywords': ['qui es tu', 'qui es-tu', 'present toi'],
            'search_in_json': ["Qui es-tu ?", "Comment tu t'appelles ?"]
        }
    }
    
    for category, info in keyword_patterns.items():
        for keyword in info['keywords']:
            if keyword in user_question_lower:
                for q in info['search_in_json']:
                    for item in questions_data:
                        if item['question_originale'] == q:
                            return item['answer'], 0.95
    return None, 0

def generate_with_groq(question: str, context: str) -> str:
    """Génère avec Groq ou fallback"""
    if not USE_GROQ or groq_client is None:
        # Fallback simple
        return "Je suis là pour t'aider avec le cancer du sein ! Pose ta question. 💗"
    
    try:
        messages = [
            {"role": "system", "content": "Tu es ANONTCHIGAN, assistante béninoise."},
            {"role": "user", "content": f"CONTEXTE:\n{context}\n\nQUESTION: {question}"}
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=200,
            temperature=0.8,
        )
        
        return response.choices[0].message.content.strip()
    except:
        return "Je suis là pour t'aider ! Reformule ta question. 💗"

# ============================================
# ENDPOINT CHAT
# ============================================

@app.post("/chat")
async def chat(query: Query):
    """Endpoint principal"""
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")
    
    logger.info(f"📥 Question: {query.question}")
    
    # Salutations
    salutations = ["bonjour", "salut", "hello"]
    if query.question.lower().strip() in salutations:
        return {
            "answer": "Akwè ! 😊 C'est ANONTCHIGAN. Tu as une question ?",
            "status": "success",
            "method": "salutation"
        }
    
    # Recherche par mots-clés
    question_lower = query.question.lower()
    keyword_answer, score = find_by_keywords(question_lower)
    if keyword_answer and score >= 0.9:
        return {
            "answer": keyword_answer,
            "status": "success",
            "method": "keyword_match"
        }
    
    # Recherche FAISS
    results = search_faiss(query.question, k=3)
    
    if not results:
        return {
            "answer": "Je n'ai pas trouvé d'info. Reformule ta question ! 💗",
            "status": "info",
            "method": "no_result"
        }
    
    best = results[0]
    similarity = best['similarity']
    
    # Seuil de décision
    if similarity >= 0.65:
        return {
            "answer": best['answer'],
            "status": "success",
            "method": "json_direct",
            "score": float(similarity)
        }
    else:
        # Génération avec contexte
        context = f"Q: {best['question']}\nR: {best['answer'][:200]}"
        generated = generate_with_groq(query.question, context)
        
        return {
            "answer": generated,
            "status": "success",
            "method": "groq_generated",
            "score": float(similarity)
        }

# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
