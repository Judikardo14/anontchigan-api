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
    page_title="ANONTCHIGAN",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# MASQUAGE DES ÉLÉMENTS STREAMLIT
# ============================================

hide_streamlit_style = """
<style>
    /* Masquer le menu hamburger */
    #MainMenu {visibility: hidden;}
    
    /* Masquer le footer "Made with Streamlit" */
    footer {visibility: hidden;}
    
    /* Masquer le header Streamlit */
    header {visibility: hidden;}
    
    /* Masquer le bouton "Deploy" */
    .stDeployButton {display: none;}
    
    /* Masquer le lien GitHub */
    .viewerBadge_container__1QSob {display: none;}
    
    /* Masquer l'icône GitHub */
    .viewerBadge_link__1S137 {display: none;}
    
    /* Masquer tous les badges */
    .stApp header {display: none;}
    
    /* Style pour cacher complètement le header */
    div[data-testid="stToolbar"] {display: none;}
    
    /* Cacher le bouton fork */
    button[title="View source on GitHub"] {display: none;}
    
    /* Masquer "Created by" */
    .css-1v0mbdj {display: none;}
    
    /* Masquer le logo Streamlit rouge en bas */
    .css-1dp5vir {display: none;}
    
    /* Masquer toute référence à Streamlit */
    a[href*="streamlit.io"] {display: none;}
    
    /* Masquer le footer complet de Streamlit */
    footer, .reportview-container .main footer {visibility: hidden;}
    
    /* Réduire l'espace en haut */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
                max_tokens=550,
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
5. Ne mentionne le nom de tes créateurs que si la question te demande de te présenter

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
        unwanted_intros = ['well','done']
        
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
                not
