import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
import gc

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
embedding_model = None
index = None
questions_data = []
USE_GROQ = False
groq_client = None

# Modèle pour les requêtes
class Query(BaseModel):
    question: str

# ============================================
# GESTIONNAIRE DE CYCLE DE VIE (NOUVEAU)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise au démarrage, nettoie à l'arrêt"""
    global embedding_model, index, questions_data, USE_GROQ, groq_client
    
    logger.info("="*50)
    logger.info("🚀 Démarrage d'ANONTCHIGAN...")
    logger.info("="*50)
    
    try:
        # FIX CRITIQUE pour Streamlit Cloud
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
        
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
        
        # 3. FIX: Utiliser un modèle compatible avec Streamlit Cloud
        logger.info("🔍 Chargement du modèle d'embeddings...")
        
        try:
            # CHANGEMENT CRITIQUE: Modèle plus stable pour Streamlit
            import torch
            torch.set_num_threads(1)  # Limite l'usage CPU
            
            embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',  # Plus stable que L3-v2
                device='cpu',
                cache_folder='/tmp/sentence_transformers'
            )
            
            # Test que ça marche
            test_embed = embedding_model.encode(
                "test", 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            logger.info(f"✓ Modèle chargé (dim={len(test_embed)})")
            
        except Exception as e:
            logger.error(f"❌ Erreur modèle: {e}")
            # Plan B: modèle encore plus léger
            try:
                logger.info("🔄 Tentative avec modèle alternatif...")
                embedding_model = SentenceTransformer(
                    'paraphrase-MiniLM-L6-v2',
                    device='cpu'
                )
                logger.info("✓ Modèle alternatif chargé")
            except Exception as e2:
                logger.error(f"❌ Échec total: {e2}")
                raise
        
        # 4. Création de l'index FAISS
        logger.info("📊 Création de l'index FAISS...")
        
        # Encoder par petits lots pour économiser la mémoire
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = embedding_model.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"✓ Index FAISS créé ({len(embeddings)} vecteurs, dim={dimension})")
        
        # Nettoyer la mémoire
        del embeddings, all_embeddings, all_texts
        gc.collect()
        
        logger.info("="*50)
        logger.info("✓ ANONTCHIGAN PRÊT !")
        logger.info(f"  - Groq: {'Activé ⚡' if USE_GROQ else 'Désactivé'}")
        logger.info(f"  - Questions: {len(questions_data)}")
        logger.info("="*50)
        
        yield  # L'application tourne ici
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}", exc_info=True)
        raise
    finally:
        # Nettoyage à l'arrêt
        logger.info("🛑 Arrêt de l'application...")
        gc.collect()

# Initialiser FastAPI avec le nouveau gestionnaire
app = FastAPI(
    title="ANONTCHIGAN API",
    description="Assistante IA béninoise pour la sensibilisation au cancer du sein",
    version="2.0.0",
    lifespan=lifespan  # NOUVEAU: Remplace @app.on_event
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "questions_loaded": len(questions_data),
        "model_dimension": embedding_model.get_sentence_embedding_dimension() if embedding_model else None
    }

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def search_faiss(query: str, k: int = 3):
    """Recherche dans FAISS"""
    if embedding_model is None or index is None:
        return []
    
    try:
        query_embedding = embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(questions_data):
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    'question': questions_data[idx]['question_originale'],
                    'answer': questions_data[idx]['answer'],
                    'similarity': float(similarity),
                    'distance': float(distances[0][i])
                })
        
        return results
    except Exception as e:
        logger.error(f"Erreur FAISS: {e}")
        return []

def find_by_keywords(user_question_lower):
    """Recherche par mots-clés"""
    keyword_patterns = {
        'identite': {
            'keywords': ['qui es tu', 'qui es-tu', 'present toi', 'présente toi'],
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
        return "Je suis là pour t'aider avec le cancer du sein ! Pose ta question. 💗"
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "Tu es ANONTCHIGAN, assistante béninoise spécialisée dans la sensibilisation au cancer du sein. Réponds de manière claire, empathique et en français."
            },
            {
                "role": "user", 
                "content": f"CONTEXTE:\n{context}\n\nQUESTION: {question}\n\nRéponds de manière claire et bienveillante."
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=250,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur Groq: {e}")
        return "Je suis là pour t'aider ! Reformule ta question. 💗"

# ============================================
# ENDPOINT CHAT
# ============================================

@app.post("/chat")
async def chat(query: Query):
    """Endpoint principal de chat"""
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")
    
    logger.info(f"📥 Question: {query.question}")
    
    # Salutations
    salutations = ["bonjour", "salut", "hello", "hey", "akwè"]
    if query.question.lower().strip() in salutations:
        return {
            "answer": "Akwè ! 😊 C'est ANONTCHIGAN. Comment puis-je t'aider aujourd'hui ?",
            "status": "success",
            "method": "salutation"
        }
    
    # Recherche par mots-clés
    question_lower = query.question.lower()
    keyword_answer, score = find_by_keywords(question_lower)
    if keyword_answer and score >= 0.9:
        logger.info("✓ Réponse par mot-clé")
        return {
            "answer": keyword_answer,
            "status": "success",
            "method": "keyword_match",
            "score": float(score)
        }
    
    # Recherche FAISS
    results = search_faiss(query.question, k=3)
    
    if not results:
        logger.warning("⚠️ Aucun résultat FAISS")
        return {
            "answer": "Je n'ai pas trouvé d'information sur ce sujet. Peux-tu reformuler ta question ? 💗",
            "status": "info",
            "method": "no_result"
        }
    
    best = results[0]
    similarity = best['similarity']
    
    logger.info(f"📊 Meilleur score: {similarity:.3f}")
    
    # Seuil de décision
    if similarity >= 0.65:
        logger.info("✓ Réponse directe du JSON")
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
        
        logger.info("✓ Réponse générée avec Groq")
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
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
