import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import random
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Initialiser FastAPI
app = FastAPI()

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

# Servir la page d'accueil
@app.get("/")
async def serve_home():
    return FileResponse("index.html")

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "ANONTCHIGAN API is running"}

# ============================================
# CONFIGURATION GROQ
# ============================================
print("🔧 Configuration de Groq...")
GROQ_API_KEY =  os.getenv("GROQ_API_KEY")  # Votre clé API Groq
USE_GROQ = False
groq_client = None

try:
    from groq import Groq
    
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Test de connexion
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.1-8b-instant",
            max_tokens=5,
        )
        USE_GROQ = True
        print("✓ Groq configuré (Llama 3.1 8B Instant)")
        print("  → Vitesse: Ultra-rapide ⚡")
        print("  → Limite: 6000 req/min (gratuit)")
    else:
        print("⚠️ Clé API Groq manquante")
        print("💡 Obtenez-en une gratuitement sur : https://console.groq.com")
        
except ImportError:
    print("❌ Module groq manquant")
    print("💡 Installez : pip install groq")
except Exception as e:
    print(f"⚠️ Groq non disponible: {str(e)[:100]}")
    USE_GROQ = False

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
print("🚀 Démarrage d'ANONTCHIGAN...")
print("Chargement des données RAG...")
try:
    with open('cancer_sein.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Fichier cancer_sein.json non trouvé")
    data = []

# Créer la base de connaissances
questions_data = []
all_texts = []
for item in data:
    questions_data.append({
        'question_originale': item['question'],
        'question_normalisee': item['question'].lower().strip(),
        'answer': item['answer']
    })
    all_texts.append(f"Question: {item['question']}\nRéponse: {item['answer']}")

print(f"✓ {len(questions_data)} questions chargées")

# ============================================
# INITIALISATION EMBEDDINGS + FAISS
# ============================================
print("🔍 Initialisation des embeddings...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Modèle d'embeddings chargé")

print("📊 Création de l'index FAISS...")
embeddings = embedding_model.encode(all_texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✓ Index FAISS créé ({len(embeddings)} vecteurs, dim={dimension})")

# ============================================
# FONCTIONS DE RECHERCHE
# ============================================

def similarity_score(str1, str2):
    """Calcule la similarité entre deux chaînes (0 à 1)"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def search_faiss(query: str, k: int = 3):
    """Recherche dans FAISS"""
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
    """Trouve une réponse basée sur des mots-clés importants"""
    keyword_patterns = {
        'identite': {
            'keywords': ['qui es tu', 'qui es-tu', 'present toi', 'qui est anontchigan'],
            'search_in_json': ["Qui es-tu ?", "Comment tu t'appelles ?"]
        },
        'nom_signification': {
            'keywords': ['signifie ton nom', 'veut dire ton nom', 'signification nom'],
            'search_in_json': ["Que signifie ton nom ?", "Pourquoi t'appelles-tu ANONTCHIGAN ?"]
        },
        'createurs': {
            'keywords': ['qui t a cree', 'créé', 'developpe par', 'qui a fait', 'créateurs'],
            'search_in_json': ["Qui t'a créé ?", "Par qui as-tu été développé ?"]
        },
        'club_ia': {
            'keywords': ['club ia', 'club d ia', 'ensgmm'],
            'search_in_json': ["C'est quoi le club d'IA de l'ENSGMM ?"]
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

# ============================================
# GÉNÉRATION AVEC GROQ (ULTRA-RAPIDE ⚡)
# ============================================

def generate_fallback_from_context(context: str) -> str:
    """Génère une réponse de secours basée sur le contexte (style béninois 100%)"""
    try:
        lines = context.split('\n')
        answers = [line.strip() for line in lines if line.strip().startswith('R:')]
        
        if answers:
            first_answer = answers[0].replace('R:', '').strip()
            if len(first_answer) > 250:
                first_answer = first_answer[:247] + "..."
            
            intro_phrases = [
                "Wê, écoute bien : ",
                "Bon, voilà ce que je peux te dire oh : ",
                "Franchement hein, ",
                "Écoute, on est ensemble : ",
                "Regarde, "
            ]
            outro_phrases = [
                " N'oublie pas de consulter un docteur hein ! Y'a rien.",
                " Va voir un professionnel sef, on ne rigole pas avec ça ! 💗",
                " Mais consulte un médecin aussi oh. On est dedans ! 🇧🇯",
                " Et va à l'hôpital faire ton check-up. Dèdè dèdè ! 💗"
            ]
            return f"{random.choice(intro_phrases)}{first_answer}{random.choice(outro_phrases)}"
        
        encouragement_phrases = [
            "C'est bien de te renseigner ! 💗 Pense à faire l'auto-examen chaque mois et va voir un docteur. La prévention ça sauve des vies même ! 🌸🇧🇯",
            "Y'a rien ! 😊 Fais ton auto-examen régulièrement et consulte un professionnel. On ne donne pas dos, on est ensemble ! 💗🇧🇯",
            "Franchement, c'est bien de s'informer ! 💗 Auto-examen + médecin = santé garantie sef ! On est là ! 🌸",
            "Tu fais bien de poser la question 🔥 N'oublie pas : auto-examen tous les mois et check-up annuel. Dèdè dèdè ! 💗🇧🇯"
        ]
        return random.choice(encouragement_phrases)
    except:
        return "Y'a pas de souci ! 😊 Fais ton auto-examen régulièrement et va voir un docteur. On est ensemble, on ne donne pas dos ! 💗🇧🇯"

def generate_with_groq(question: str, context: str) -> str:
    """Génère une réponse avec Groq (ULTRA-RAPIDE ⚡)"""
    
    if not USE_GROQ or groq_client is None:
        print("⚠️ Groq non configuré. Utilisation du contexte direct.")
        return generate_fallback_from_context(context)
    
    try:
        # Optimiser le contexte
        context_lines = context.split('\n')[:8]
        context_short = '\n'.join(context_lines)
        
        if len(context_short) > 1000:
            context_short = context_short[:1000] + "..."
        
        # Messages pour Groq (format Chat)
        messages = [
            {
                "role": "system",
                "content": """Tu es ANONTCHIGAN 💗, une assistante IA BÉNINOISE 100% authentique, joviale et chaleureuse, spécialisée dans la sensibilisation au cancer du sein au Bénin.

PERSONNALITÉ BÉNINOISE AUTHENTIQUE:
- Tu parles le vrai français du Bénin avec toutes les expressions locales
- Vocabulaire Fon/Yoruba: "akwè" (salut), "atassa/atassssaaa" (étonnement/ce qui dépasse l'entendement), "dèdè dèdè" (doucement doucement), "wê" (regarde), "gbê" (vie), "gnon" (argent)
- Expressions béninoises AUTHENTIQUES: 
  * "on est ensemble" / "on est là" / "on est dedans"
  * "y'a rien" / "y'a pas de souci" / "ça ne fait rien sef"
  * "dèdè dèdè" (doucement doucement)
  * "c'est comment?" / "ça dit quoi?" / "on va faire comment?"
  * "tu m'as où là?" / "tu me dépasses oh!"
  * "franchement hein" / "sérieusement là"
  * "même là" / "sef" / "là même"
  * "gnin gnin gnin" (bavardages inutiles)
  * "kpayo" (mensonge/arnaque)
  * "donner dos" (abandonner)
  * "mon frère" / "ma sœur" / "ma grande" / "tantine" / "mon gars"
- Références locales: Cotonou, Porto-Novo, Dantokpa, zemidjans, gari, aloko, akassa, la vie béninoise
- Tu es chaleureuse comme les vraies Béninoises: accueillante, directe, maternelle, parfois taquine
- Ton humour est typiquement béninois (on rigole ensemble, on se taquine gentiment)

STYLE DE CONVERSATION NATUREL:
- NE répète JAMAIS "Bonjour" en début de réponse 
- Réponds DIRECTEMENT et naturellement comme au marché ou dans la rue à Cotonou
- Sois concise (2-4 phrases) mais vraiment chaleureuse
- Utilise UNIQUEMENT les infos du contexte fourni
- Si hors sujet: "Ahh non, gnin gnin gnin là, c'est pas mon domaine ! 😄 Moi c'est cancer du sein. On fait comment?"
- Parfois utilise des métaphores béninoises (gari, marché, zemidjan, etc.) si pertinent

RÈGLES:
- IMPORTANT: N'utilise "mon frère/mon gars" QUE si tu es CERTAIN que c'est un homme (question sur prostate, "je suis un homme", etc.)
- N'utilise "ma sœur/ma grande/tantine" QUE si tu es CERTAINE que c'est une femme
- Pour les questions sur auto-examen des seins, cancer du sein, etc.: reste NEUTRE, ne désigne pas le genre
- Formules neutres: "Écoute", "Wê", "Regarde", "Franchement", sans désignation de genre
- Utilise "atassa!" ou "atassssaaa!" pour exprimer l'étonnement, ce qui dépasse l'entendement
- "Kpayo!" si quelqu'un croit de fausses infos
- Encourage naturellement, sans être lourde ni donner dos
- Emojis: 💗, 🌸, 😊, 🇧🇯, ✨, 🔥
- Cite Cotonou, centres de santé béninois, vie locale si ça colle au contexte"""
            },
            {
                "role": "user",
                "content": f"""CONTEXTE de ma base de connaissances:
{context_short}

QUESTION de l'utilisateur: {question}

IMPORTANT: Ne dis PAS "mon frère" ou "ma sœur" sauf si tu es CERTAIN du genre de la personne. Pour les questions sur seins/cancer du sein, reste NEUTRE.
Evite d'être trop chaleureuse et d'employer des mots comme "Atassa" fin de phrase. C'est pour les début et à employer 
une seule fois en cas de grande surprise ou étonnement. Les mots comme wê doivent être emplyé rarement dans les réponses 
et une fois en passant.
Réponds DIRECTEMENT (sans dire "Bonjour", "Salut" ou formule d'intro). Sois naturelle, chaleureuse et concise:"""
            }
        ]
        
        print(f"🤖 Génération avec Groq (Llama 3.1)...")
        
        # Appel à Groq - ULTRA RAPIDE
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Modèle le plus rapide
            messages=messages,
            max_tokens=200,  # Réduit pour forcer la concision
            temperature=0.8,  # Augmenté pour plus de créativité/humour
            top_p=0.9,
        )
        
        # Extraire la réponse
        answer = response.choices[0].message.content.strip()
        
        # Nettoyer les formules indésirables
        unwanted_starts = [
            "bonjour", "salut", "hello", "coucou", "bonsoir", "akwè",
            "chère femme", "cher ami", "ma chère",
            "bonjour à tous", "salut à tous"
        ]
        
        for phrase in unwanted_starts:
            # Retirer si c'est au tout début
            if answer.lower().startswith(phrase):
                answer = answer[len(phrase):].strip()
                # Retirer la ponctuation qui suit
                if answer and answer[0] in [',', '!', '.', ':']:
                    answer = answer[1:].strip()
                # Capitaliser la première lettre
                if answer:
                    answer = answer[0].upper() + answer[1:]
        
        # Vérifier que la réponse n'est pas vide
        if not answer or len(answer) < 10:
            print("⚠️ Réponse trop courte, utilisation du fallback")
            return generate_fallback_from_context(context)
        
        print(f"✓ Réponse générée avec Groq ({len(answer)} caractères)")
        return answer
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erreur Groq: {error_msg}")
        
        # Messages d'erreur explicites
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            print("💡 Limite de taux atteinte (rare avec Groq).")
        elif "api key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print("💡 Problème d'authentification.")
            print("   Vérifiez votre clé sur https://console.groq.com")
        elif "invalid" in error_msg.lower():
            print("💡 Clé API invalide. Obtenez-en une sur https://console.groq.com")
        
        # Fallback sur le contexte
        return generate_fallback_from_context(context)

# ============================================
# ENDPOINT CHAT PRINCIPAL
# ============================================

@app.post("/chat")
async def chat(query: Query):
    print(f"\n📥 Question: {query.question}")
    
    # 1. SALUTATIONS
    salutations = ["bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
    question_lower = query.question.lower().strip()
    
    if any(salut == question_lower for salut in salutations):
        responses = [
            "Akwè ! 😊 C'est ANONTCHIGAN ici 💗 Ça dit quoi ? Tu veux savoir quelque chose ?",
            "Eyy ! 👋 On est ensemble hein ! Tu as une question sur le cancer du sein ? 🇧🇯",
            "Ça va ou bien ? 🌸 Moi c'est ANONTCHIGAN. C'est comment, qu'est-ce qui t'amène ?",
            "Akwè akwè ! 💗 Pose ta question là, on va gérer ça dodo dodo 😊",
            "Eyy ma grande/mon frère ! ✨ ANONTCHIGAN à ton service. Y'a pas de souci, on est là ! 🇧🇯"
        ]
        return {
            "answer": random.choice(responses),
            "status": "success",
            "method": "salutation"
        }
    
    # 2. RECHERCHE PAR MOTS-CLÉS
    keyword_answer, keyword_score = find_by_keywords(question_lower)
    if keyword_answer and keyword_score >= 0.9:
        print(f"✅ Réponse par mot-clé (score: {keyword_score:.2f})")
        return {
            "answer": keyword_answer,
            "status": "success",
            "method": "keyword_match",
            "score": float(keyword_score)
        }
    
    # 3. RECHERCHE FAISS
    print("🔍 Recherche FAISS...")
    faiss_results = search_faiss(query.question, k=3)
    
    if not faiss_results:
        print("⚠️ Aucun résultat FAISS")
        
        # Vérifier si c'est hors sujet
        off_topic_keywords = [
            'football', 'météo', 'politique', 'recette', 'cuisine', 
            'film', 'musique', 'jeux', 'sport', 'voyage', 'ordinateur',
            'programmation', 'code', 'app', 'téléphone'
        ]
        
        if any(keyword in question_lower for keyword in off_topic_keywords):
            off_topic_responses = [
                "Ahh non, gnin gnin gnin là, c'est pas mon domaine ! 😄 Moi c'est cancer du sein que je gère. Ça dit quoi, on parle de ça ? 💗🇧🇯",
                "Eyy, tu m'as où franchement ? 😅 Mon truc c'est le cancer du sein oh. On fait comment, tu veux savoir quoi ? 🌸",
                "Wê ! Là même là, je ne suis pas dedans 😊 Mais pour le cancer du sein, je suis là ! Pose ta question sef ! 💗",
                "Là tu me dépasses oh ! 🤣 Reviens au cancer du sein, c'est là je suis forte. On est ensemble ? 🇧🇯"
            ]
            return {
                "answer": random.choice(off_topic_responses),
                "status": "info",
                "method": "off_topic_redirect"
            }
        
        no_result_responses = [
            "Hmm, je n'ai pas bien compris ça oh 🤔 Reformule dèdè ou pose ta question sur le cancer du sein. Y'a rien, on est là ! 💗",
            "Wê, là même je ne trouve pas l'info 😅 Essaie de reformuler ou demande autre chose sur le cancer du sein. On va gérer ! 🌸",
            "Franchement, ça là je ne connais pas trop 🤔 Mais pose-moi sur le cancer du sein, je suis dedans ! On est ensemble ! 💗🇧🇯"
        ]
        return {
            "answer": random.choice(no_result_responses),
            "status": "info",
            "method": "no_result"
        }
    
    best_result = faiss_results[0]
    similarity = best_result['similarity']
    
    print(f"📊 Meilleure similarité FAISS: {similarity:.3f}")
    print(f"   Question trouvée: {best_result['question']}")
    
    # 4. DÉCISION : JSON vs GÉNÉRATION
    SIMILARITY_THRESHOLD = 0.65
    
    if similarity >= SIMILARITY_THRESHOLD:
        # HAUTE SIMILARITÉ → Réponse directe du JSON
        print(f"✅ Haute similarité ({similarity:.2f}) → Réponse du JSON")
        return {
            "answer": best_result['answer'],
            "status": "success",
            "method": "json_direct",
            "score": float(similarity),
            "matched_question": best_result['question']
        }
    
    else:
        # FAIBLE SIMILARITÉ → Génération avec Groq
        print(f"🤖 Faible similarité ({similarity:.2f}) → Génération Groq")
        
        # Préparer le contexte (top 2-3 résultats)
        context_parts = []
        for i, result in enumerate(faiss_results[:3], 1):
            # Limiter chaque réponse
            answer_truncated = result['answer']
            if len(answer_truncated) > 250:
                answer_truncated = answer_truncated[:247] + "..."
            context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
        
        context = "\n\n".join(context_parts)
        
        # Générer avec Groq
        generated_answer = generate_with_groq(query.question, context)
        
        print(f"📤 Réponse générée ({len(generated_answer)} caractères)")
        return {
            "answer": generated_answer,
            "status": "success",
            "method": "groq_generated",
            "score": float(similarity),
            "context_used": len(faiss_results[:3])
        }

print("\n" + "="*50)
print("✓ ANONTCHIGAN prêt (RAG Hybride + Groq)")
print("  - Recherche: FAISS + Mots-clés")
print(f"  - Génération: {'Groq ⚡ (Ultra-rapide)' if USE_GROQ else 'Contexte direct ⚠️'}")
print(f"  - Seuil similarité: 0.65")
if not USE_GROQ:
    print("\n⚠️  Pour activer la génération Groq :")
    print("   1. Créez un compte sur https://console.groq.com")
    print("   2. Obtenez votre API key (gratuit)")
    print("   3. Définissez : export GROQ_API_KEY='votre_clé'")
    print("   4. Installez : pip install groq")
print("="*50 + "\n")

if __name__ == "__main__":
    print("💫 Démarrage du serveur...")
    print("🌐 Interface: http://localhost:8000\n")
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)