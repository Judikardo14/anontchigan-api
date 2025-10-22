"""
API REST pour ANONTCHIGAN - Sans problème CORS
Déployer avec: uvicorn main:app --host 0.0.0.0 --port 8000
Ou sur: Railway, Render, Vercel, etc.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import json
import os
import logging
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANONTCHIGAN_API")

app = FastAPI(
    title="ANONTCHIGAN API",
    description="API de sensibilisation au cancer du sein au Bénin",
    version="3.0.0"
)

# DÉSACTIVATION COMPLÈTE DU CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accepte toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Tous les headers
)

class Config:
    SIMILARITY_THRESHOLD = 0.75
    MAX_CONTEXT_LENGTH = 1000
    FAISS_RESULTS_COUNT = 3
    MIN_ANSWER_LENGTH = 30

# ============================================
# MODÈLES PYDANTIC
# ============================================

class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = "default"
    history: Optional[List[Dict]] = []

class ResponseModel(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None

# ============================================
# SERVICES (identiques au code Streamlit)
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
                return
            
            self.client = Groq(api_key=api_key)
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.1-8b-instant",
                max_tokens=5,
            )
            self.available = True
            logger.info("✓ Groq initialisé")
        except Exception as e:
            logger.warning(f"Groq non disponible: {str(e)}")
    
    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        if not self.available:
            raise RuntimeError("Groq non disponible")
        
        context_short = context[:Config.MAX_CONTEXT_LENGTH]
        
        system_prompt = f"""Tu es ANONTCHIGAN, assistante IA pour le cancer du sein au Bénin.
CONTEXTE: {context_short}
Réponds de manière claire, empathique avec emojis 💗 🌸"""
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )
        
        answer = response.choices[0].message.content.strip()
        return answer

class RAGService:
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self.embedding_model = None
        self.index = None
        self._load_data(data_file)
        self._initialize_embeddings()
    
    def _load_data(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            self.questions_data.append({
                'question_originale': item['question'],
                'answer': item['answer']
            })
        logger.info(f"✓ {len(self.questions_data)} questions chargées")
    
    def _initialize_embeddings(self):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        all_texts = [f"Q: {item['question_originale']} R: {item['answer']}" for item in self.questions_data]
        self.embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings).astype('float32')
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logger.info("✓ FAISS initialisé")
    
    def search(self, query: str, k: int = Config.FAISS_RESULTS_COUNT) -> List[Dict]:
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
                    'similarity': similarity
                })
        return results

# ============================================
# INITIALISATION DES SERVICES
# ============================================

groq_service = GroqService()
rag_service = RAGService()

# Stockage en mémoire des conversations
conversation_storage = []

# ============================================
# ENDPOINTS API
# ============================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "name": "ANONTCHIGAN API",
        "version": "3.0.0",
        "endpoints": {
            "POST /api/ask": "Poser une question (JSON body)",
            "GET /api/ask": "Poser une question (URL params)",
            "GET /api/history": "Récupérer l'historique complet",
            "DELETE /api/history": "Effacer l'historique",
            "GET /health": "Vérifier l'état de l'API"
        },
        "cors_enabled": True
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "groq_available": groq_service.available,
        "rag_loaded": rag_service.index is not None,
        "conversations_count": len(conversation_storage)
    }

@app.post("/api/ask")
async def ask_question_post(request: QuestionRequest):
    """
    Poser une question via POST avec JSON body
    
    Body exemple:
    {
        "question": "Symptômes du cancer du sein ?",
        "user_id": "user123",
        "history": []
    }
    """
    try:
        result = process_question(
            request.question, 
            request.history or [], 
            request.user_id
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ask")
async def ask_question_get(
    question: str = Query(..., description="Question à poser"),
    user_id: str = Query("default", description="ID utilisateur")
):
    """
    Poser une question via GET avec paramètres URL
    
    Exemple: /api/ask?question=Symptômes+cancer+sein&user_id=user123
    """
    try:
        result = process_question(question, [], user_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(user_id: Optional[str] = None):
    """
    Récupérer l'historique complet ou filtré par user_id
    
    Exemple: /api/history?user_id=user123
    """
    if user_id:
        filtered = [c for c in conversation_storage if c.get("user_id") == user_id]
        return {
            "success": True,
            "count": len(filtered),
            "user_id": user_id,
            "conversations": filtered
        }
    
    return {
        "success": True,
        "count": len(conversation_storage),
        "conversations": conversation_storage
    }

@app.delete("/api/history")
async def clear_history():
    """Effacer tout l'historique"""
    conversation_storage.clear()
    return {
        "success": True,
        "message": "Historique effacé"
    }

@app.get("/api/export")
async def export_history():
    """
    Exporter l'historique en JSON téléchargeable
    """
    from fastapi.responses import Response
    
    json_data = json.dumps(conversation_storage, ensure_ascii=False, indent=2)
    
    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=anontchigan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )

# ============================================
# LOGIQUE DE TRAITEMENT
# ============================================

def process_question(question: str, history: List[Dict], user_id: str = "default"):
    """Traite une question et retourne la réponse"""
    
    # Salutations
    salutations = ["cc", "bonjour", "salut", "hello", "yo"]
    if question.lower().strip() in salutations:
        answer = "Bonjour ! Je suis ANONTCHIGAN 💗 Comment puis-je vous aider ?"
        method = "salutation"
        score = None
    else:
        # Recherche FAISS
        faiss_results = rag_service.search(question)
        
        if not faiss_results:
            answer = "Je recommande de consulter un professionnel de santé au Bénin. 💗"
            method = "no_result"
            score = None
        else:
            best_result = faiss_results[0]
            similarity = best_result['similarity']
            
            if similarity >= Config.SIMILARITY_THRESHOLD:
                # Réponse directe
                answer = best_result['answer']
                method = "json_direct"
                score = float(similarity)
            else:
                # Génération avec Groq
                context_parts = []
                for i, result in enumerate(faiss_results[:3], 1):
                    context_parts.append(f"{i}. Q: {result['question']}\n   R: {result['answer'][:200]}")
                context = "\n\n".join(context_parts)
                
                try:
                    if groq_service.available:
                        answer = groq_service.generate_response(question, context, history)
                        method = "groq_generated"
                    else:
                        answer = "Veuillez consulter un professionnel. 💗"
                        method = "fallback"
                except Exception:
                    answer = "Consultez un médecin spécialisé au Bénin. 🌸"
                    method = "error_fallback"
                
                score = float(similarity)
    
    # Créer l'entrée de conversation
    conversation_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "method": method,
        "similarity_score": score
    }
    
    # Sauvegarder dans l'historique
    conversation_storage.append(conversation_entry)
    
    # Retourner la réponse
    return {
        "success": True,
        "data": conversation_entry
    }

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
