# 🏦 QUESTRAG - Banking QUEries and Support system via Trained Reinforced RAG

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3.1-blue.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed on HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/eeshanyaj/questrag-backend)

> An intelligent banking chatbot powered by **Retrieval-Augmented Generation (RAG)** and **Reinforcement Learning (RL)** to provide accurate, context-aware responses to Indian banking queries while optimizing token costs.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Performance Metrics](#performance-metrics)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Status](#status)
- [Links](#links)

---

## 🎯 Overview
QUESTRAG is an **advanced banking chatbot** designed to revolutionize customer support in the Indian banking sector. By combining **Retrieval-Augmented Generation (RAG)** with **Reinforcement Learning (RL)**, the system intelligently decides when to fetch external context from a knowledge base and when to respond directly, **reducing token costs by up to 31%** while maintaining high accuracy.

### Problem Statement
Existing banking chatbots suffer from:
- ❌ Limited response flexibility (rigid, rule-based systems)
- ❌ Poor handling of informal/real-world queries
- ❌ Lack of contextual understanding
- ❌ High operational costs due to inefficient token usage
- ❌ Low user satisfaction and trust

### Solution
QUESTRAG addresses these challenges through:
- ✅ **Domain-specific RAG** trained on 19,000+ banking queries / support data
- ✅ **RL-optimized policy network** (BERT-based) for smart context-fetching decisions
- ✅ **Fine-tuned retriever model** (E5-base-v2) using InfoNCE + Triplet Loss
- ✅ **Groq LLM with HuggingFace fallback** for reliable, fast responses
- ✅ **Full-stack web application** with modern UI/UX and JWT authentication

---

## 🌟 Key Features

### 🤖 Intelligent RAG Pipeline
- **FAISS-powered retrieval** for fast similarity search across 19,352 documents
- **Fine-tuned embedding model** (`e5-base-v2`) trained on English + Hinglish paraphrases
- **Context-aware response generation** using Llama 3 models (8B & 70B) via Groq

### 🧠 Reinforcement Learning System
- **BERT-based policy network** (`bert-base-uncased`) for FETCH/NO_FETCH decisions
- **Reward-driven optimization** (+2.0 accurate, +0.5 needed fetch, -0.5 incorrect)
- **31% token cost reduction** via optimized retrieval

### 🎨 Modern Web Interface
- **React 18 + Vite** with Tailwind CSS
- **Real-time chat**, conversation history, JWT authentication
- **Responsive design** for desktop and mobile

### 🔐 Enterprise-Ready Backend
- **FastAPI + MongoDB Atlas** for scalable async operations
- **JWT authentication** with secure password hashing (bcrypt)
- **Multi-provider LLM** (Groq → HuggingFace automatic fallback)
- **Deployed on HuggingFace Spaces** with Docker containerization

---

## 🏗️ System Architecture

<p align="center">
  <img src="./assets/system.png" alt="System Architecture Diagram" width="750"/>
</p>

### 🔄 Workflow
1. **User Query** → FastAPI receives query via REST API
2. **Policy Decision** → BERT-based RL model decides FETCH or NO_FETCH
3. **Conditional Retrieval** → If FETCH → Retrieve top-5 docs from FAISS using E5-base-v2
4. **Response Generation** → Llama 3 (via Groq) generates final answer
5. **Evaluation & Logging** → Logged in MongoDB + reward-based model update

---

## 🔄 Sequence Diagram

<p align="center">
  <img src="./assets/sequence_diagram.png" alt="Sequence Diagram" width="750"/>
</p>

---

## 🛠️ Technology Stack

### **Frontend**
- ⚛️ React 18.3.1 + Vite 5.4.2
- 🎨 Tailwind CSS 3.4.1
- 🔄 React Context API + Axios + React Router DOM

### **Backend**
- 🚀 FastAPI 0.104.1
- 🗄️ MongoDB Atlas + Motor (async driver)
- 🔑 JWT Auth + Passlib (bcrypt)
- 🤖 PyTorch 2.9.1, Transformers 4.57, FAISS 1.13.0
- 💬 Groq (Llama 3.1 8B Instant / Llama 3.3 70B Versatile)
- 🎯 Sentence Transformers 5.1.2

### **Machine Learning**
- 🧠 **Policy Network**: BERT-base-uncased (trained with RL)
- 🔍 **Retriever**: E5-base-v2 (fine-tuned with InfoNCE + Triplet Loss)
- 📊 **Vector Store**: FAISS (19,352 documents)

### **Deployment**
- 🐳 Docker (HuggingFace Spaces)
- 🤗 HuggingFace Hub (model storage)
- ☁️ MongoDB Atlas (cloud database)
- 🌐 Python 3.12 + uvicorn

---

## ⚙️ Installation

### 🧩 Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB Atlas account (or local MongoDB 6.0+)
- Groq API key (or HuggingFace token)

### 🔧 Backend Setup (Local Development)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your credentials (see Configuration section)

# Build FAISS index (one-time setup)
python build_faiss_index.py

# Start backend server
uvicorn app.main:app --reload --port 8000
```

### 💻 Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
# Update VITE_API_URL to point to your backend

# Start dev server
npm run dev
```

---

## ⚙️ Configuration

### 🔑 Backend `.env` (Key Parameters)

| **Category**      | **Key**                          | **Example / Description**                        |
|-------------------|----------------------------------|--------------------------------------------------|
| Environment       | `ENVIRONMENT`                    | `development` or `production`                    |
| MongoDB           | `MONGODB_URI`                    | `mongodb+srv://user:pass@cluster.mongodb.net/`   |
| Authentication    | `SECRET_KEY`                     | Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
|                   | `ALGORITHM`                      | `HS256`                                          |
|                   | `ACCESS_TOKEN_EXPIRE_MINUTES`    | `1440` (24 hours)                                |
| Groq API          | `GROQ_API_KEY_1`                 | Your primary Groq API key                        |
|                   | `GROQ_API_KEY_2`                 | Secondary key (optional)                         |
|                   | `GROQ_API_KEY_3`                 | Tertiary key (optional)                          |
|                   | `GROQ_CHAT_MODEL`                | `llama-3.1-8b-instant`                           |
|                   | `GROQ_EVAL_MODEL`                | `llama-3.3-70b-versatile`                        |
| HuggingFace       | `HF_TOKEN_1`                     | HuggingFace token (fallback LLM)                 |
|                   | `HF_MODEL_REPO`                  | `eeshanyaj/questrag_models` (for model download) |
| Model Paths       | `POLICY_MODEL_PATH`              | `app/models/best_policy_model.pth`               |
|                   | `RETRIEVER_MODEL_PATH`           | `app/models/best_retriever_model.pth`            |
|                   | `FAISS_INDEX_PATH`               | `app/models/faiss_index.pkl`                     |
|                   | `KB_PATH`                        | `app/data/final_knowledge_base.jsonl`            |
| Device            | `DEVICE`                         | `cpu` or `cuda`                                  |
| RAG Params        | `TOP_K`                          | `5` (number of documents to retrieve)            |
|                   | `SIMILARITY_THRESHOLD`           | `0.5` (minimum similarity score)                 |
| Policy Network    | `CONFIDENCE_THRESHOLD`           | `0.7` (policy decision confidence)               |
| CORS              | `ALLOWED_ORIGINS`                | `http://localhost:5173` or `*`                   |

### 🌐 Frontend `.env`

```bash
# Local development
VITE_API_URL=http://localhost:8000

# Production (HuggingFace Spaces)
VITE_API_URL=https://eeshanyaj-questrag-backend.hf.space
```

---

## 🚀 Usage

### 🖥️ Local Development

#### Start Backend Server

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### Start Frontend Dev Server

```bash
cd frontend
npm run dev
```

- **Frontend**: http://localhost:5173

### 🌐 Production (HuggingFace Spaces)

**Backend API**:
- **Base URL**: https://eeshanyaj-questrag-backend.hf.space
- **API Docs**: https://eeshanyaj-questrag-backend.hf.space/docs
- **Health Check**: https://eeshanyaj-questrag-backend.hf.space/health

**Frontend** (Coming Soon):
- Will be deployed on Vercel/Netlify

---

## 📁 Project Structure

```
questrag/
│
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── auth.py              # Auth endpoints (register, login)
│   │   │   └── chat.py              # Chat endpoints
│   │   ├── core/
│   │   │   ├── llm_manager.py       # Groq + HF LLM orchestration
│   │   │   └── security.py          # JWT & password hashing
│   │   ├── ml/
│   │   │   ├── policy_network.py    # RL Policy model (BERT)
│   │   │   └── retriever.py         # E5-base-v2 retriever
│   │   ├── db/
│   │   │   ├── mongodb.py           # MongoDB connection
│   │   │   └── repositories/        # User & conversation repos
│   │   ├── services/
│   │   │   └── chat_service.py      # Orchestration logic
│   │   ├── models/
│   │   │   ├── best_policy_model.pth      # Trained policy network
│   │   │   ├── best_retriever_model.pth   # Fine-tuned retriever
│   │   │   └── faiss_index.pkl            # FAISS vector store
│   │   ├── data/
│   │   │   └── final_knowledge_base.jsonl # 19,352 Q&A pairs
│   │   ├── config.py                # Settings & env vars
│   │   └── main.py                  # FastAPI app entry point
│   ├── Dockerfile                   # Docker config for HF Spaces
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── components/              # UI Components
    │   ├── context/                 # Auth Context
    │   ├── pages/                   # Login, Register, Chat
    │   ├── services/api.js          # Axios Client
    │   ├── App.jsx
    │   └── main.jsx
    ├── package.json
    └── .env
```

---

## 📊 Datasets

### 1. Final Knowledge Base
- **Size**: 19,352 question-answer pairs
- **Categories**: 15 banking categories
- **Intents**: 22 unique intents (ATM, CARD, LOAN, ACCOUNT, etc.)
- **Source**: Combination of:
  - Bitext Retail Banking Dataset (Hugging Face)
  - RetailBanking-Conversations Dataset
  - Manually curated FAQs from SBI, ICICI, HDFC, Yes Bank, Axis Bank

### 2. Retriever Training Dataset
- **Size**: 11,655 paraphrases
- **Source**: 1,665 unique FAQs from knowledge base
- **Paraphrases per FAQ**:
  - 4 English paraphrases
  - 2 Hinglish paraphrases
  - Original FAQ
- **Training**: InfoNCE Loss + Triplet Loss with E5-base-v2

### 3. Policy Network Training Dataset
- **Size**: 182 queries from 6 chat sessions
- **Format**: (state, action, reward) tuples
- **Actions**: FETCH (1) or NO_FETCH (0)
- **Rewards**: +2.0 (correct), +0.5 (needed fetch), -0.5 (incorrect)

---

## 📈 Performance Metrics

*Coming soon: Detailed performance metrics including accuracy, response time, token cost reduction, and user satisfaction scores.*

---

## 📚 API Documentation

### Authentication

#### Register

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:**

```json
{
  "message": "User registered successfully",
  "user_id": "507f1f77bcf86cd799439011"
}
```

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "john_doe",
  "password": "securepassword123"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

---

### Chat

#### Send Message

```http
POST /api/v1/chat/
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What are the interest rates for home loans?",
  "conversation_id": "optional-session-id"
}
```

**Response:**

```json
{
  "response": "Current home loan interest rates range from 8.5% to 9.5% per annum...",
  "conversation_id": "abc123",
  "metadata": {
    "policy_action": "FETCH",
    "retrieval_score": 0.89,
    "documents_retrieved": 5,
    "llm_provider": "groq"
  }
}
```

#### Get Conversation History

```http
GET /api/v1/chat/conversations/{conversation_id}
Authorization: Bearer <token>
```

**Response:**

```json
{
  "conversation_id": "abc123",
  "messages": [
    {
      "role": "user",
      "content": "What are the interest rates?",
      "timestamp": "2025-11-28T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Current rates are...",
      "timestamp": "2025-11-28T10:30:05Z",
      "metadata": {
        "policy_action": "FETCH"
      }
    }
  ]
}
```

#### List All Conversations

```http
GET /api/v1/chat/conversations
Authorization: Bearer <token>
```

#### Delete Conversation

```http
DELETE /api/v1/chat/conversation/{conversation_id}
Authorization: Bearer <token>
```

---

## 🚀 Deployment

### HuggingFace Spaces (Backend)

The backend is deployed on HuggingFace Spaces using Docker:

1. **Models are stored** on HuggingFace Hub: `eeshanyaj/questrag_models`
2. **On first startup**, models are automatically downloaded from HF Hub
3. **Docker container** runs FastAPI with uvicorn on port 7860
4. **Environment secrets** are securely managed in HF Space settings

**Deployment Steps:**

```bash
# 1. Upload models to HuggingFace Hub
huggingface-cli upload eeshanyaj/questrag_models \
  app/models/best_policy_model.pth \
  models/best_policy_model.pth

# 2. Push backend code to HF Space
git remote add space https://huggingface.co/spaces/eeshanyaj/questrag-backend
git push space main

# 3. Add environment secrets in HF Space Settings
# (MongoDB URI, Groq keys, JWT secret, etc.)
```

### Frontend Deployment (Vercel/Netlify)

```bash
# Build for production
npm run build

# Deploy to Vercel
vercel --prod

# Update .env.production with backend URL
VITE_API_URL=https://eeshanyaj-questrag-backend.hf.space
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint + Prettier for JavaScript/React
- Write comprehensive docstrings and comments
- Add unit tests for new features
- Update documentation accordingly

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

### Research Inspiration
- **Main Paper**: "Optimizing Retrieval Augmented Generation for Domain-Specific Chatbots with Reinforcement Learning" (AAAI 2024)
- **Additional References**:
  - "Evaluating BERT-based Rewards for Question Generation with RL"
  - "Self-Reasoning for Retrieval-Augmented Language Models"

### Open Source Resources
- [RL-Self-Improving-RAG](https://github.com/subrata-samanta/RL-Self-Improving-RAG)
- [ARENA](https://github.com/ren258/ARENA)
- [RAGTechniques](https://github.com/NirDiamant/RAGTechniques)
- [Financial-RAG-From-Scratch](https://github.com/cse-amarjeet/Financial-RAG-From-Scratch)

### Datasets
- [Bitext Retail Banking Dataset](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset)
- [RetailBanking-Conversations](https://huggingface.co/datasets/oopere/RetailBanking-Conversations)

### Technologies
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [HuggingFace](https://huggingface.co/)
- [Groq](https://groq.com/)
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)

---

## 📞 Contact

**Eeshanya Amit Joshi**  
📧 [Email](mailto:eeshanyajoshi@gmail.com)    
💼 [LinkedIn](https://www.linkedin.com/in/eeshanyajoshi/)

---

## 📈 Status

### ✅ **Backend Deployed & Live!**
- 🚀 Backend API running on [HuggingFace Spaces](https://eeshanyaj-questrag-backend.hf.space)
- 📚 API Documentation available at [/docs](https://eeshanyaj-questrag-backend.hf.space/docs)
- 💚 Health status: [Check here](https://eeshanyaj-questrag-backend.hf.space/health)

### 🚧 **Frontend Deployment - Coming Soon!**
- Will be deployed on Vercel/Netlify
- Stay tuned for full application link! ❤️

---

## 🔗 Links

- **Live Backend API:** https://eeshanyaj-questrag-backend.hf.space
- **API Documentation:** https://eeshanyaj-questrag-backend.hf.space/docs
- **Health Check:** https://eeshanyaj-questrag-backend.hf.space/health
- **HuggingFace Space:** https://huggingface.co/spaces/eeshanyaj/questrag-backend
- **Model Repository:** https://huggingface.co/eeshanyaj/questrag_models
- **Research Paper:** [AAAI 2024 Workshop](https://arxiv.org/abs/2401.06800)

---

<p align="center">✨ Made with ❤️ for the Banking Industry ✨</p>
<p align="center">Powered by HuggingFace 🤗| Groq ⚡| MongoDB 🍃| Docker 🐳| </p>
