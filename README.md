# 🎤 MockMate-AI Mock Interview Assistant

An AI-powered voice-based interview simulation platform that evaluates user responses and generates a detailed PDF report — built with Flask, LangChain, Whisper, and ChromaDB.

---

## 🚀 Features

- Upload your resume (PDF)
- Auto-generated mock interview questions (technical, behavioral, and role-specific)
- Voice-based Q&A with real-time transcription using OpenAI Whisper
- AI evaluation of answers with scores, feedback, and categories
- Interactive frontend with progress tracking
- Final PDF report with scores, feedback, and summary

---

## 📁 Project Structure

```
.
├── app.py                        # Flask backend with API endpoints
├── mock_interview_agent.py      # Core logic: resume parsing, interview generation, evaluation
├── main.js                      # Frontend interaction logic (audio recording, session control)
├── index.html                   # UI template (Bootstrap based)
├── requirements.txt             # Python dependencies
├── templates/
│   └── index.html               # HTML entry point for Flask
├── static/
│   └── main.js                  # Frontend JavaScript
├── uploads/                     # Uploaded resumes
├── reports/                     # Generated PDF reports
├── chroma_db/                   # Resume vector DB (auto-generated)
└── eval_db/                     # Evaluation vector DB (auto-generated)
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mock-interview-assistant.git
cd mock-interview-assistant
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file:

```
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Install FFmpeg

Download and install FFmpeg, then add it to your system PATH.

📽️ [Install guide](https://youtu.be/GYdhqmy_Nt8?si=6Rmr8Po4vWqNTFYg)

---

## ▶️ Running the App

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 Powered By

- 🧩 **LangChain + LangGraph** — dynamic agent workflows
- 🗂️ **ChromaDB** — vector storage for resumes and evaluations
- 🧠 **Whisper** — real-time audio transcription
- 📄 **ReportLab** — export structured PDF reports
- 🎙️ **Pyttsx3 & Pyaudio** — voice input/output

---

## 📌 Future Improvements

- Role selection before question generation
- Scoring visualization (e.g., radar/spider charts)
- Support for multilingual interviews
- Option to switch to FastAPI backend

---

## 📃 License

MIT License — free to use and modify!

---

## ✨ Demo Screenshot

![screenshot](https://user-images.githubusercontent.com/demo-screenshot.png) <!-- Replace with actual image if available -->

---

## 🙌 Contribute

Pull requests welcome! Please open an issue first if you want to add a feature or fix a bug.
