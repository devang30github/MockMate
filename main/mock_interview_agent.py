
# mock_interview_agent.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
import fitz  # PyMuPDF
import os
import shutil
import ast
import re
from datetime import datetime
from dotenv import load_dotenv
import pyaudio
import wave
import whisper
import keyboard

load_dotenv()

CHROMA_DB_DIR = "chroma_db"
EVAL_DB_DIR = "eval_db"
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="meta-llama/llama-4-maverick:free",
)

embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

# ---------- DB Utilities ----------

def clear_chroma_db():
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    if os.path.exists(EVAL_DB_DIR):
        shutil.rmtree(EVAL_DB_DIR)

# ---------- Resume Handling ----------

def extract_text_from_resume(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def store_resume_in_chroma(text):
    docs = [Document(page_content=text, metadata={"source": "resume"})]
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="resume_collection"
    )

# ---------- LangGraph Interview Flow ----------

def get_resume_context(state):
    query = "Extract relevant details for generating interview questions."
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="resume_collection"
    )
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"resume_context": context}

def generate_questions(state):
    context = state["resume_context"]
    question_prompt = rf"""
You're an AI interview coach. Based on the following resume, generate 1 mock interview question per category:

{context}

Respond strictly as a Python dictionary:
{{
  "technical": ["..."],
  "behavioral": ["..."],
  "role_specific": ["..."]
}}
"""
    response = llm.invoke([HumanMessage(content=question_prompt)])
    return {"mock_questions": response.content}

def create_interview_flow(state):
    raw_questions = state["mock_questions"]
    try:
        match = re.search(r"(\{.*\})", raw_questions, re.DOTALL)
        dict_str = match.group(1)
        question_dict = ast.literal_eval(dict_str)
    except Exception:
        question_dict = {"General": [raw_questions.strip()]}

    opening = "Tell me about yourself."
    closing = "Do you have any questions for me?"

    all_questions = [opening]
    for category in question_dict.values():
        all_questions.extend(category)
    all_questions.append(closing)

    return {"interview_flow": all_questions}

from typing import TypedDict

class ResumeState(TypedDict):
    resume_context: str
    mock_questions: str
    interview_flow: list[str]

def create_langgraph_workflow():
    graph = StateGraph(state_schema=ResumeState)
    graph.add_node("GetResumeContext", get_resume_context)
    graph.add_node("GenerateQuestions", generate_questions)
    graph.add_node("CreateInterviewFlow", create_interview_flow)

    graph.set_entry_point("GetResumeContext")
    graph.add_edge("GetResumeContext", "GenerateQuestions")
    graph.add_edge("GenerateQuestions", "CreateInterviewFlow")
    graph.set_finish_point("CreateInterviewFlow")

    return graph.compile()

# ---------- AGENTS ----------

class QuestionSpeakerAgent:
    def __init__(self, rate=150, volume=1.0):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak_question(self, question: str):
        self.engine.say(question)
        self.engine.runAndWait()

class AnswerEvaluatorAgent:
    def __init__(self, model_name="base", rate=44100, channels=1, chunk=1024):
        self.model = whisper.load_model(model_name)
        self.rate = rate
        self.channels = channels
        self.chunk = chunk

    def record_audio(self) -> str:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=self.channels,
                            rate=self.rate, input=True, frames_per_buffer=self.chunk)

        frames = []
        while not keyboard.is_pressed('enter'):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

        return filename

    def transcribe_audio(self, filepath: str) -> str:
        result = self.model.transcribe(filepath)
        return result["text"]

    def evaluate_answer(self, question: str, answer: str) -> dict:
        prompt = rf"""
You are an AI interview coach. Evaluate the following:

Question: {question}
Answer: {answer}

Respond only with a Python dictionary:
{{
  "score": 1-10,
  "feedback": "...",
  "category": "technical"/"behavioral"/"role_specific"
}}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def record_transcribe_evaluate(self, question: str):
        filepath = self.record_audio()
        try:
            transcription = self.transcribe_audio(filepath)
            evaluation = self.evaluate_answer(question, transcription)
        finally:
            os.remove(filepath)

        return {
            "transcription": transcription,
            "evaluation": evaluation
        }

class FinalFeedbackAgent:
    def __init__(self):
        self.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=EVAL_DB_DIR,
            collection_name="interview_evaluations"
        )

    def generate_summary(self):
        evaluations = self.vectorstore.similarity_search("all evaluations", k=5)
        combined = "\n\n".join(doc.page_content for doc in evaluations)

        prompt = rf"""
You are a professional AI interviewer. Based on the following evaluations:

{combined}

Write:
- Performance Summary
- Strengths
- Weaknesses
- Suggestions

Respond clearly.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

# ---------- Evaluation Storage ----------

def store_evaluation_in_chroma(evaluation_data):
    docs = []
    for item in evaluation_data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}\nEvaluation: {item['evaluation']}"
        docs.append(Document(page_content=content))

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=EVAL_DB_DIR,
        collection_name="interview_evaluations"
    )

# ---------- PDF Export ----------

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def export_pdf_report(results, final_feedback, filename="mock_interview_report.pdf"):
    def safe_parse_eval(eval_str):
        try:
            eval_str = eval_str.strip()
            if eval_str.startswith("```"):
                eval_str = re.sub(r"```(?:\w+)?", "", eval_str).strip()
            return ast.literal_eval(eval_str)
        except Exception:
            return {"score": 0, "feedback": "Invalid format", "category": "unknown"}

    doc = SimpleDocTemplate(filename, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Mock Interview Report", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    parsed_evals = [safe_parse_eval(r['evaluation']) for r in results]
    avg_score = sum([e['score'] for e in parsed_evals]) / len(parsed_evals)
    story.append(Paragraph(f"Average Score: {avg_score:.2f}/10", styles['Heading2']))

    for i, (r, eval_data) in enumerate(zip(results, parsed_evals), 1):
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Q{i}: {r['question']}", styles['Heading3']))
        story.append(Paragraph(f"Answer: {r['answer']}", styles['Normal']))
        story.append(Paragraph(f"Score: {eval_data['score']}", styles['Normal']))
        story.append(Paragraph(f"Feedback: {eval_data['feedback']}", styles['Normal']))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Final Feedback Summary:", styles['Heading2']))
    for line in final_feedback.split("\n"):
        story.append(Paragraph(line.strip(), styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
