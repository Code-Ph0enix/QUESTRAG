---
title: QUESTRAG Backend
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# QUESTRAG Banking Chatbot Backend

FastAPI backend for QUESTRAG - Banking RAG Chatbot with Reinforcement Learning.

## Features
- 🤖 RAG Pipeline with FAISS
- 🧠 RL-based Policy Network
- ⚡ Groq (Llama 3) + HuggingFace fallback
- 🔐 JWT Authentication
- 📊 MongoDB Atlas

## API Documentation
Visit `/docs` for interactive Swagger UI.

## Endpoints
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/chat/` - Send message (requires auth)
- `GET /health` - Health check
