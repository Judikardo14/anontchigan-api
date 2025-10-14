# 🩷 ANONTCHIGAN API — Chatbot Béninois pour la Sensibilisation au Cancer du Sein 🇧🇯

**ANONTCHIGAN** est une API intelligente développée avec **FastAPI**, **FAISS** et **Sentence Transformers**, spécialisée dans la sensibilisation au **cancer du sein** au Bénin 🇧🇯.
Elle combine la recherche sémantique (RAG) et la génération de texte via **Llama 3.1 (Groq)**, avec une personnalité 100% béninoise : chaleureuse, naturelle et éducative.

---

## 🚀 Fonctionnalités

* 🔍 Recherche de similarité avec **FAISS**
* ⚡ Génération ultra-rapide avec **Groq (Llama 3.1)**
* 💬 Style de réponse **authentiquement béninois**
* 🧠 Données médicales fiables (issues de `cancer_sein.json`)
* 🔒 Gestion sécurisée des clés API via `.env`
* 🌐 Compatible avec tout front-end (React, Vue, etc.)

---

## 🧰 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/<ton-utilisateur>/anontchigan-api.git
cd anontchigan-api
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Créer un fichier `.env`

Crée un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=ta_cle_api_groq
```

👉 Pour obtenir ta clé : [https://console.groq.com](https://console.groq.com)

---

## ▶️ Lancer le serveur

```bash
uvicorn app:app --reload
```

API disponible sur :
➡️ [http://localhost:8000](http://localhost:8000)

Endpoint de test :
➡️ [http://localhost:8000/health](http://localhost:8000/health)

---

## 💬 Exemple d’utilisation

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Quels sont les signes du cancer du sein ?"}'
```

Réponse typique 👇

```json
{
  "answer": "Wê, souvent on remarque une boule ou un écoulement anormal du sein. Mais va voir un docteur hein ! 💗",
  "status": "success",
  "method": "json_direct"
}
```

---

## 📁 Structure du projet

```
anontchigan-api/
│
├── app.py                 # Code principal FastAPI
├── cancer_sein.json       # Base de connaissances
├── requirements.txt       # Dépendances Python
├── .env.example           # Exemple de configuration d’environnement
├── .gitignore             # Fichiers à ignorer par Git
└── README.md              # Documentation du projet
```

---

## 🧑🏽‍💻 Auteur

**Projet ANONTCHIGAN 💗**
Développé par   Judicaël Karol DOBOEVI, 
                Hornel Ursus GBAGUIDI, 
                Abel Kocou KPOKOUTA, 
                Josaphat ADJELE

Membres du **Club IA ENSGMM 🇧🇯**

---

## ⚖️ Licence

Ce projet est distribué sous licence **MIT** — vous pouvez l’utiliser, le modifier et le partager librement.

---

> 💡 *« La connaissance, c’est la prévention. Et la prévention, c’est la vie ! »* — ANONTCHIGAN 🌸🇧🇯
