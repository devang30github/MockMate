# mock_interview_agent.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

import fitz  # PyMuPDF
from os import getenv
from dotenv import load_dotenv
import shutil
import os
import ast  # For parsing LLM dictionary output
import re

import pyttsx3
import speech_recognition as sr
import whisper
import keyboard
import pyaudio
import wave
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch

# ----------------- ENV SETUP -----------------
load_dotenv()
CHROMA_DB_DIR = "chroma_db"
EVAL_DB_DIR = "eval_db"
HF_MODEL_NAME = getenv("HF_MODEL_NAME")

# ----------------- LLM + PROMPT -----------------
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="meta-llama/llama-4-maverick:free",
)

# ----------------- EMBEDDINGS -----------------
embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

# ----------------- CLEAR CHROMA DB -----------------
def clear_chroma_db():
    if os.path.exists(CHROMA_DB_DIR):
        print("🧹 Clearing old ChromaDB data...")
        shutil.rmtree(CHROMA_DB_DIR)
        print("✅ ChromaDB reset complete.")
    if os.path.exists(EVAL_DB_DIR):
        print("🧹 Clearing old Evaluation DB...")
        shutil.rmtree(EVAL_DB_DIR)
        print("✅ Evaluation DB reset complete.")

# ----------------- RESUME TEXT EXTRACTION -----------------
def extract_text_from_resume(file_path):
    print(f"📄 Extracting text from: {file_path}")
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------- STORE RESUME IN CHROMA -----------------
def store_resume_in_chroma(text):
    docs = [Document(page_content=text or "", metadata={"source": "resume"})]
    print("📦 Storing resume in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="resume_collection"
    )
    print("✅ Stored successfully.")

# ----------------- LANGGRAPH NODES -----------------
def get_resume_context(state):
    query = "Extract all relevant details for generating interview questions (skills, experience, projects)"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="resume_collection"
    )
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"resume_context": context}

def generate_questions(state):
    context = state["resume_context"]
    question_prompt = f"""
You're an AI interview coach. Based on the following resume, generate 1 per category mock interview questions in the form of a Python dictionary with three categories: 
- Technical
- Behavioral
- Role-specific

**Instructions**:
1. Provide **only** a Python dictionary, formatted correctly as (curly braces).
2. Do **not** include any explanations, markdown, or additional text. 
3. The dictionary should contain the following categories:
   - "technical": List of technical questions.
   - "behavioral": List of behavioral questions.
   - "role_specific": List of role-specific questions.

Here’s the resume context:

{context}
"""

    response = llm.invoke([HumanMessage(content=question_prompt)])
    return {"mock_questions": response.content}

def create_interview_flow(state):
    raw_questions = state["mock_questions"]

    try:
        match = re.search(r"(\{.*\})", raw_questions, re.DOTALL)
        if not match:
            raise ValueError("No dictionary found in response.")

        dict_str = match.group(1)
        question_dict = ast.literal_eval(dict_str)

    except Exception as e:
        print(f"⚠️ Failed to parse mock questions: {e}")
        question_dict = {"General": [raw_questions.strip()]}

    opening = "Tell me about yourself."
    closing = "Do you have any questions for me?"

    all_questions = [opening]
    for category in question_dict.values():
        all_questions.extend(category)
    all_questions.append(closing)

    return {"interview_flow": all_questions}

# ----------------- AGENT 1 -----------------
def ask_question_tts(question):
    engine = pyttsx3.init()
    engine.say(question)
    engine.runAndWait()

# ----------------- AGENT 2 -----------------
def listen_to_answer():
    print("\n🎙️ Recording... Press [Enter] when you're done answering.")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    try:
        while not keyboard.is_pressed('enter'):
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("⛔ Recording manually stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wave_path = f"recording_{timestamp}.wav"

    with wave.open(wave_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Audio saved to {wave_path}")

    try:
        print("🧠 Transcribing your answer...")
        model = whisper.load_model("base")
        result = model.transcribe(wave_path)
        return result["text"]
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return "[Unintelligible]"
    finally:
        if os.path.exists(wave_path):
            os.remove(wave_path)
            print(f"🧹 Deleted temp file: {wave_path}")

def evaluate_answer(question, answer):
    prompt = f"""
You are an AI interview coach. Evaluate the candidate's answer below.

Question: "{question}"
Answer: "{answer}"

Provide a Python dictionary with:
- "score": (1-10),
- "feedback": A short paragraph.
- "category": "technical", "behavioral", or "role_specific"

Respond with only the dictionary.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ----------------- STORE AGENT 2 EVALS -----------------
def store_evaluation_in_chroma(evaluation_data, persist_dir=EVAL_DB_DIR):
    docs = []
    for item in evaluation_data:
        content = f"""Question: {item['question']}
Answer: {item['answer']}
Evaluation: {item['evaluation']}"""
        docs.append(Document(page_content=content))

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="interview_evaluations"
    )
    print("✅ Stored evaluations in vector DB.")

# ----------------- AGENT 3 -----------------
def agent3_generate_final_feedback():
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=EVAL_DB_DIR,
        collection_name="interview_evaluations"
    )
    evaluations = vectorstore.similarity_search("all evaluations", k=5)
    combined = "\n\n".join([doc.page_content for doc in evaluations])

    prompt = f"""
You are a professional AI interview coach. Based on the following interview session (questions, answers, evaluations), give an overall feedback.

Structure your response into:
- Summary of performance
- Strengths
- Weaknesses
- Suggestions for improvement

Here’s the session:

{combined}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ----------------- INTERVIEW EXECUTION -----------------
def run_interview(llm, questions):
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n🔹 Question {i}: {question}")
        ask_question_tts(question)

        answer = listen_to_answer()
        print(f"🗣️ Transcribed Answer: {answer}")

        feedback = evaluate_answer(question, answer)
        print(f"✅ Evaluation: {feedback}")

        results.append({
            "question": question,
            "answer": answer,
            "evaluation": feedback
        })

    store_evaluation_in_chroma(results)
    return results

# ----------------- LANGGRAPH -----------------
from typing import TypedDict

class ResumeState(TypedDict):
    resume_context: str
    mock_questions: str
    interview_flow: list[str]

def create_langgraph_workflow():
    graph = StateGraph(state_schema=ResumeState)

    graph.add_node("GetResumeContext", RunnableLambda(get_resume_context))
    graph.add_node("GenerateQuestions", RunnableLambda(generate_questions))
    graph.add_node("CreateInterviewFlow", RunnableLambda(create_interview_flow))

    graph.set_entry_point("GetResumeContext")
    graph.add_edge("GetResumeContext", "GenerateQuestions")
    graph.add_edge("GenerateQuestions", "CreateInterviewFlow")
    graph.set_finish_point("CreateInterviewFlow")

    return graph.compile()

def export_pdf_report(results, final_feedback, filename="mock_interview_report.pdf"):
    def safe_parse_eval(eval_str):
        eval_str = eval_str.strip()
        # Remove markdown code fencing if present
        if eval_str.startswith("```"):
            eval_str = re.sub(r"```(?:\w+)?", "", eval_str).strip()
        try:
            return ast.literal_eval(eval_str)
        except Exception as e:
            print(f"⚠️ Failed to parse evaluation string: {e}")
            return {"score": 0, "feedback": "Invalid format", "category": "unknown"}

    doc = SimpleDocTemplate(filename, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("📝 <b>Mock Interview Report</b>", styles['Title']))
    story.append(Spacer(1, 0.3 * inch))

    parsed_evals = [safe_parse_eval(r['evaluation']) for r in results]
    avg_score = sum([e['score'] for e in parsed_evals]) / len(parsed_evals)
    story.append(Paragraph(f"<b>Average Score:</b> {avg_score:.2f}/10", styles['Heading2']))
    story.append(Spacer(1, 0.2 * inch))

    for i, (r, eval_data) in enumerate(zip(results, parsed_evals), 1):
        story.append(Paragraph(f"<b>Q{i}:</b> {r['question']}", styles['Heading3']))
        story.append(Paragraph(f"<b>Answer:</b> {r['answer']}", styles['Normal']))
        story.append(Paragraph(f"<b>Score:</b> {eval_data['score']}", styles['Normal']))
        story.append(Paragraph(f"<b>Feedback:</b> {eval_data['feedback']}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("<b>🏁 Final Feedback Summary:</b>", styles['Heading2']))
    for line in final_feedback.strip().split("\n"):
        story.append(Paragraph(line.strip(), styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    print(f"📄 PDF report saved as: {filename}")
# ----------------- MAIN FLOW -----------------
def process_resume(file_path):
    clear_chroma_db()

    text = extract_text_from_resume(file_path)
    store_resume_in_chroma(text)

    print("🧠 Generating mock interview questions...\n")
    workflow = create_langgraph_workflow()
    result = workflow.invoke({})

    print("🎤 Starting Mock Interview...\n")
    interview_results=run_interview(llm, result["interview_flow"])

    print("🧠 Generating final feedback from Agent 3...\n")
    final_feedback = agent3_generate_final_feedback()
    print(f"🏁 Final Feedback:\n{final_feedback}")

    export_pdf_report(interview_results, final_feedback)

# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    resume_path = "sample_resume.pdf"
    process_resume(resume_path)
