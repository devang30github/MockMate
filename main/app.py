# app.py

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import io

# Import from your updated agent script
from mock_interview_agent import (
    clear_chroma_db,
    extract_text_from_resume,
    store_resume_in_chroma,
    create_langgraph_workflow,
    QuestionSpeakerAgent,
    AnswerEvaluatorAgent,
    FinalFeedbackAgent,
    store_evaluation_in_chroma,
    export_pdf_report
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

interview_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

# -------- Upload Resume --------

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
    file.save(filepath)

    return jsonify({'filepath': filepath}), 200

# -------- Process Resume and Generate Questions --------

@app.route('/process', methods=['POST'])
def process_resume():
    data = request.json
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        clear_chroma_db()
        text = extract_text_from_resume(filepath)
        store_resume_in_chroma(text)
        workflow = create_langgraph_workflow()
        result = workflow.invoke({})
        questions = result.get("interview_flow")

        if not questions:
            return jsonify({'error': 'Failed to generate questions'}), 500

        return jsonify({'result': {'questions': questions}}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------- Start Interview Session --------

@app.route('/interview/start', methods=['POST'])
def start_interview():
    data = request.json
    questions = data.get('questions', [])

    if not questions:
        return jsonify({'error': 'No questions provided'}), 400

    session_id = str(uuid.uuid4())
    interview_sessions[session_id] = {
        'questions': questions,
        'current_index': 0,
        'answers': [],
        'evaluations': []
    }

    return jsonify({
        'session_id': session_id,
        'total_questions': len(questions),
        'current_question': {'text': questions[0]}
    }), 200

# -------- Speak Question --------

@app.route('/speak_question/<session_id>', methods=['GET'])
def speak_question(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = interview_sessions[session_id]
    current_idx = session['current_index']

    if current_idx >= len(session['questions']):
        return jsonify({'error': 'No more questions'}), 400

    question_text = session['questions'][current_idx]

    speaker = QuestionSpeakerAgent()
    speaker.speak_question(question_text)

    return jsonify({'message': 'Question spoken successfully'}), 200

# -------- Submit Audio Answer --------

@app.route('/interview/audio_answer/<session_id>', methods=['POST'])
def submit_audio_answer(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.save(temp_audio.name)
    temp_audio.close()

    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio.name)
        transcription = result["text"]

        session = interview_sessions[session_id]
        current_idx = session['current_index']
        question = session['questions'][current_idx]

        evaluator = AnswerEvaluatorAgent()
        evaluation_result = evaluator.evaluate_answer(question, transcription)

        session['answers'].append(transcription)
        session['evaluations'].append(evaluation_result)

        is_completed = current_idx >= len(session['questions']) - 1
        session['current_index'] += 1

        os.unlink(temp_audio.name)

        return jsonify({
            'transcription': transcription,
            'evaluation': {'evaluation': evaluation_result, 'is_completed': is_completed}
        }), 200

    except Exception as e:
        if os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)
        return jsonify({'error': str(e)}), 500

# -------- Get Next Question --------

@app.route('/interview/next_question/<session_id>', methods=['GET'])
def get_next_question(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = interview_sessions[session_id]
    current_idx = session['current_index']

    if current_idx >= len(session['questions']):
        return jsonify({'error': 'No more questions'}), 400

    return jsonify({'question': session['questions'][current_idx]}), 200

# -------- Check Interview Status --------

@app.route('/interview/status/<session_id>', methods=['GET'])
def get_interview_status(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = interview_sessions[session_id]
    total = len(session['questions'])
    current = session['current_index']
    remaining = total - current

    return jsonify({
        'status': 'completed' if remaining == 0 else 'in_progress',
        'total_questions': total,
        'current_index': current,
        'remaining_questions': remaining
    }), 200

# -------- Finalize Interview --------

@app.route('/interview/finalize/<session_id>', methods=['POST'])
def finalize_interview(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = interview_sessions[session_id]

    evaluation_data = []
    for i, (q, a, e) in enumerate(zip(
        session['questions'][:len(session['answers'])],
        session['answers'],
        session['evaluations']
    )):
        evaluation_data.append({
            'question': q,
            'answer': a,
            'evaluation': e
        })

    store_evaluation_in_chroma(evaluation_data)

    agent3 = FinalFeedbackAgent()
    final_feedback = agent3.generate_summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"interview_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)

    export_pdf_report(evaluation_data, final_feedback, filename=report_path)

    return jsonify({
        'final_feedback': final_feedback,
        'report_url': f"/report/{report_filename}"
    }), 200

# -------- Download Report --------

@app.route('/report/<filename>', methods=['GET'])
def download_report(filename):
    return send_file(os.path.join(REPORTS_FOLDER, filename), as_attachment=True)

# -------- Session Cleanup --------

def cleanup_sessions():
    while True:
        time.sleep(3600)
        current_time = time.time()

        expired_sessions = []
        for session_id, session in list(interview_sessions.items()):
            if 'last_access' in session and current_time - session['last_access'] > 86400:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del interview_sessions[session_id]

        for folder in [UPLOAD_FOLDER, REPORTS_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and (time.time() - os.path.getmtime(file_path)) > 86400:
                    os.remove(file_path)

if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()
    app.run(debug=True, host='0.0.0.0', port=5000)
