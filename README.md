# Lovable Clone — AI Website Generator

A full-stack AI-powered web application that generates complete websites from a single text prompt — inspired by Lovable.dev.

---

## Features

- **Prompt-to-Website:** Enter any prompt and get a fully generated, runnable website
- **Multi-Agent AI Pipeline:** Specialized agents handle planning, architecture, coding, running, and error fixing
- **Auto Error Resolution:** Runtime errors are automatically detected and resolved
- **Code Editing:** Edit the generated code directly in the UI
- **Code Download:** Download the entire generated project as a zip
- **Project Isolation:** Each prompt runs in its own sandboxed workspace
- **Pastel UI:** Playful, pastel-themed React frontend for a lovable experience

---

## How to Run

### Backend (Python)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the backend server:
   ```bash
   python backend/main.py
   ```

### Frontend (React + Vite)

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the frontend:
   ```bash
   npm run dev
   ```
3. Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Architecture & Workflow

```
User Prompt
    │
    ▼
┌─────────────┐
│   Planner   │  → Converts prompt into a structured engineering plan
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Architect  │  → Breaks plan into file-by-file implementation steps
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Coder    │  → Generates actual code for each step using LLMs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Runner    │  → Installs dependencies and starts the generated app
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Resolver   │  → Auto-detects and fixes runtime/code errors (with retries)
└─────────────┘
       │
       ▼
  Generated Website
```

---

## What Has Been Built

### 1. Backend (Python, Flask, LangChain, LangGraph)

- **Flask API Server** — Exposes endpoints for prompt submission and result retrieval (`backend/main.py`)
- **Agent Graph** — LangGraph-based orchestration of all agents (`backend/agent/graph.py`)
- **Planner Agent** — Converts user prompt into a structured engineering plan (features, files, tech stack)
- **Architect Agent** — Breaks the plan into detailed implementation steps per file
- **Coder Agent** — Executes each step, generating code via LLMs with custom file/shell tools
- **Runner Agent** — Installs dependencies, starts the generated app, verifies it runs
- **Resolver Agent** — Diagnoses and fixes errors automatically, retrying up to a configurable limit
- **State Management** — Pydantic models and TypedDicts to track plan, tasks, errors, and history (`backend/agent/states.py`)
- **Custom Tools** — File I/O, shell command execution, process management (`backend/agent/tools.py`)
- **Sandboxing** — Each project runs in an isolated workspace directory (`backend/agent/sandbox.py`)
- **Web Crawling** — Context gathering via crawl4ai (`backend/agent/crawl_4ai.py`)
- **LLM Support** — Groq, Ollama (Qwen2.5-coder), and Gemini (Google)

### 2. Frontend (React, Vite, CSS)

- **Prompt Input Form** — Styled input to enter and submit prompts (`frontend/PromptInput.jsx`)
- **Result Display** — Shows generated code and app output (`frontend/App.jsx`)
- **Code Editor** — Inline editing of generated code
- **Download Button** — Download the generated project
- **Pastel Theme** — Custom CSS with pastel colors, playful fonts, and heart icons (`frontend/style.css`)

---

## Project Structure

```
lovable_clone/
├── backend/
│   ├── main.py                  # Flask API server
│   └── agent/
│       ├── graph.py             # LangGraph agent pipeline
│       ├── states.py            # State models (Pydantic)
│       ├── tools.py             # Custom LLM tools
│       ├── prompts.py           # Agent system prompts
│       ├── sandbox.py           # Project isolation
│       ├── crawl_4ai.py         # Web crawling utility
│       └── agent_workspace/     # Generated project outputs
├── frontend/
│   ├── App.jsx                  # Main React app
│   ├── PromptInput.jsx          # Prompt input component
│   ├── style.css                # Pastel theme styles
│   ├── index.jsx                # Entry point
│   └── vite.config.js           # Vite config
└── README.md
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.11+, Flask, LangChain, LangGraph, Pydantic |
| LLMs | Groq, Ollama (Qwen2.5-coder), Google Gemini |
| Frontend | React, Vite, CSS |

---

Made with ❤️ for lovable experiences!
