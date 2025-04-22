# app.py

from flask import Flask, request, jsonify, send_file,render_template
from flask_cors import CORS
import os
import uuid
import tempfile
import threading
import time
from werkzeug.utils import secure_filename
from datetime import datetime

# Import from your existing script
from mock_interview_agent import (
    extract_text_from_resume,
    store_resume_in_chroma,
    create_langgraph_workflow,
    evaluate_answer,
    agent3_generate_final_feedback,
    export_pdf_report,
    store_evaluation_in_chroma,
    clear_chroma_db
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Store active interview sessions
interview_sessions = {}
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(filepath)
        return jsonify({'filepath': filepath}), 200
    
    return jsonify({'error': 'Invalid file format'}), 400


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

        print("🧪 Workflow result:", result)

        questions = result.get("interview_flow")
        if not questions or not isinstance(questions, list):
            return jsonify({'error': 'Failed to generate valid interview questions'}), 500

        return jsonify({'result': {'questions': questions}}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/interview/start', methods=['POST'])
def start_interview():
    data = request.json
    questions = data.get('questions', [])
    
    if not questions:
        return jsonify({'error': 'No questions provided'}), 400
    
    # Create a new interview session
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
        'current_question': {
            'text': questions[0]
        }
    }), 200

@app.route('/speak_question/<session_id>', methods=['GET'])
def speak_question(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = interview_sessions[session_id]
    current_idx = session.get('current_index', 0)

    # Just in case the index is too high
    if current_idx >= len(session['questions']):
        return jsonify({'error': 'No more questions'}), 400

    question_text = session['questions'][current_idx]

    # Use a unique temp filename to avoid audio caching
    temp_audio = tempfile.NamedTemporaryFile(suffix=f'_{current_idx}.wav', delete=False)
    temp_audio.close()

    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.save_to_file(question_text, temp_audio.name)
    engine.runAndWait()

    return send_file(temp_audio.name, mimetype='audio/wav')

@app.route('/interview/submit_answer/<session_id>', methods=['POST'])
def submit_answer(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.json
    answer = data.get('answer', '')
    
    if not answer:
        return jsonify({'error': 'No answer provided'}), 400
    
    session = interview_sessions[session_id]
    current_idx = session['current_index']
    question = session['questions'][current_idx]
    
    # Evaluate the answer
    try:
        evaluation_result = evaluate_answer(question, answer)
        
        # Store the response
        session['answers'].append(answer)
        session['evaluations'].append(evaluation_result)
        
        # Check if this is the last question
        is_completed = current_idx >= len(session['questions']) - 1
        
        # Update the current index for next time
        #session['current_index'] = min(current_idx + 1, len(session['questions']) - 1)
        session['current_index'] = current_idx + 1
        
        return jsonify({
            'evaluation': {
                'evaluation': evaluation_result,
                'is_completed': is_completed
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/interview/audio_answer/<session_id>', methods=['POST'])
def submit_audio_answer(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Save audio to temp file
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.save(temp_audio.name)
    temp_audio.close()
    
    try:
        # Import here to avoid loading the model until necessary
        import whisper
        
        # Transcribe audio
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio.name)
        transcription = result["text"]
        
        # Process the transcribed answer
        session = interview_sessions[session_id]
        current_idx = session['current_index']
        question = session['questions'][current_idx]
        
        # Evaluate the answer
        evaluation_result = evaluate_answer(question, transcription)
        
        # Store the response
        session['answers'].append(transcription)
        session['evaluations'].append(evaluation_result)
        
        # Check if this is the last question
        is_completed = current_idx >= len(session['questions']) - 1
        
        # Update the current index for next time
        session['current_index'] = current_idx + 1
        
        # Clean up temp file
        os.unlink(temp_audio.name)
        
        return jsonify({
            'transcription': transcription,
            'evaluation': {
                'evaluation': evaluation_result,
                'is_completed': is_completed
            }
        }), 200
    
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)
        return jsonify({'error': str(e)}), 500

@app.route('/interview/next_question/<session_id>', methods=['GET'])
def get_next_question(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = interview_sessions[session_id]
    current_idx = session['current_index']
    
    if current_idx >= len(session['questions']):
        return jsonify({'error': 'No more questions'}), 400
    
    return jsonify({
        'question': session['questions'][current_idx]
    }), 200

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


@app.route('/interview/finalize/<session_id>', methods=['POST'])
def finalize_interview(session_id):
    if session_id not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = interview_sessions[session_id]
    
    # Prepare data for storage
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
    
    # Store evaluations in vector db
    store_evaluation_in_chroma(evaluation_data)
    
    # Generate final feedback
    final_feedback = agent3_generate_final_feedback()
    
    # Generate PDF report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"interview_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    
    export_pdf_report(evaluation_data, final_feedback, filename=report_path)
    
    # Return feedback and link to report
    return jsonify({
        'final_feedback': final_feedback,
        'report_url': f"/report/{report_filename}"
    }), 200

@app.route('/report/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(REPORTS_FOLDER, filename), as_attachment=True)


# Clean up expired sessions periodically
def cleanup_sessions():
    while True:
        time.sleep(3600)  # Run every hour
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in interview_sessions.items():
            if 'last_access' in session and current_time - session['last_access'] > 86400:  # 24 hours
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del interview_sessions[session_id]


if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

