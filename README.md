MockMate – AI-Powered Voice-Based Mock Interview Platform

Developed an AI-powered mock interview platform that conducts end-to-end voice-based interviews using LangChain, Whisper, and OpenRouter LLMs.

Implemented a multi-agent architecture:

Agent 1: Asks personalized questions (TTS) based on parsed resume data and ChromaDB vector search.

Agent 2: Listens to spoken answers, transcribes them using Whisper, and evaluates responses with real-time scoring and feedback.

Agent 3: Analyzes evaluation history from vector DB and generates comprehensive final feedback.

Integrated PDF report export summarizing candidate performance, question-wise feedback, and improvement tips using reportlab.

Enabled resume parsing, semantic question generation, vector storage, and interactive voice-based UX using pyaudio, pyttsx3, and keyboard.

Designed for aspiring professionals to simulate realistic interviews and receive actionable, AI-driven guidance.

Tech Stack: Python, LangChain, Whisper, PyMuPDF, ChromaDB, ReportLab, OpenRouter, HuggingFace, TTS/STT, PyAudio
