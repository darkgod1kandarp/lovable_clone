## Lovable Kind of App

This project is a lovable, full-stack web application that lets you generate websites from your prompt. It features a playful, pastel-themed React frontend and a powerful AI-driven Python backend.

### Features
- Pastel colors, playful fonts, and heart icons for a lovable look
- Input your prompt and generate a website automatically
- Backend uses LangChain, Groq, and custom agents to plan, architect, and code
- Frontend displays results in a beautiful, welcoming interface

### How to Run

#### Backend (Python)
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Start the backend server:
	```bash
	python backend/main.py
	```

#### Frontend (React)
1. Install dependencies:
	```bash
	cd frontend
	npm install
	```
2. Start the frontend server:
	```bash
	npm run start
	```
3. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Usage
Enter your prompt in the input box and submit. The backend will process your prompt, generate a plan, break it into tasks, and create code. The frontend will display the result in a lovable style.

---

---

## Project Progress & Implementation Details

### Overview
This project is a full-stack AI-powered web application that generates websites from user prompts. It features a playful, pastel-themed React frontend and a Python backend that leverages advanced AI agents for planning, coding, and error resolution.

### What Has Been Done So Far

#### 1. **Backend (Python, Flask, LangChain, LangGraph, Groq, Ollama, Gemini)**
- **API Server:** Built with Flask, exposes endpoints for prompt submission and testing.
- **Agent Architecture:**
	- **Planner Agent:** Converts user prompt into a structured engineering plan (features, files, tech stack).
	- **Architect Agent:** Breaks the plan into detailed implementation steps, specifying what to build in each file.
	- **Coder Agent:** Executes each implementation step, generating code using LLMs and custom tools (file operations, command execution).
	- **Runner Agent:** Installs dependencies, starts the generated app, and checks if it runs successfully.
	- **Resolver Agent:** Diagnoses and fixes errors (coding or runtime), retrying up to a limit.
- **State Management:** Uses Pydantic models and TypedDicts to track plan, tasks, errors, and chat history.
- **Tooling:** Custom tools for file I/O, running shell commands, process management, and project isolation.
- **LLM Integration:** Supports Groq (GPT-OSS), Ollama (Qwen2.5-coder), and Gemini (Google) for flexible model selection.
- **Project Isolation:** Each user prompt/project is handled in a unique workspace directory for safe file operations.

#### 2. **Frontend (React, Vite, CSS)**
- **UI/UX:**
	- Pastel color palette, playful fonts, and heart icons for a lovable look.
	- Responsive layout with a welcoming header and main content area.
- **Prompt Input:**
	- Users enter prompts in a styled input form.
	- Submits prompt to backend and displays generated results.
- **Styling:**
	- Custom CSS for buttons, forms, and output display.
	- Theming for a consistent, lovable experience.

#### 3. **Architecture & Workflow**
- **Prompt Flow:**
	1. User enters a prompt in the frontend.
	2. Frontend sends the prompt to the backend API.
	3. Backend agents (planner → architect → coder → runner → resolver) process the prompt, generate code, and handle errors.
	4. Results are returned to the frontend and displayed to the user.
- **Error Handling:**
	- Automated retries and error resolution for both code generation and runtime issues.
- **Extensibility:**
	- Modular agent design allows for easy addition of new agent types or tools.

#### 4. **Technologies Used**
- **Backend:** Python 3.11+, Flask, LangChain, LangGraph, Groq, Ollama, Gemini, Pydantic
- **Frontend:** React, Vite, CSS (custom pastel theme)

---
Made with ❤️ for lovable experiences!
