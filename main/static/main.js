// 🔗 Same code as before with fixes
// 👉 Replaces your existing /static/main.js

const API_BASE_URL = "http://localhost:5000";

let sessionId = null;
let currentQuestionIndex = 0;
let totalQuestions = 0;
let mediaRecorder = null;
let audioChunks = [];

document.addEventListener("DOMContentLoaded", function () {
  const uploadSection = document.getElementById("uploadSection");
  const processingSection = document.getElementById("processingSection");
  const interviewSection = document.getElementById("interviewSection");
  const feedbackSection = document.getElementById("feedbackSection");

  const resumeUploadForm = document.getElementById("resumeUploadForm");
  const currentQuestion = document.getElementById("currentQuestion");
  const questionCounter = document.getElementById("questionCounter");
  const answerText = document.getElementById("answerText");
  const submitAnswerBtn = document.getElementById("submitAnswer");
  const startRecordingBtn = document.getElementById("startRecording");
  const stopRecordingBtn = document.getElementById("stopRecording");
  const recordingStatus = document.getElementById("recordingStatus");

  const evaluationCard = document.getElementById("evaluationCard");
  const answerScore = document.getElementById("answerScore");
  const answerCategory = document.getElementById("answerCategory");
  const answerFeedback = document.getElementById("answerFeedback");
  const nextQuestionBtn = document.getElementById("nextQuestion");

  const finalFeedbackContent = document.getElementById("finalFeedbackContent");
  const downloadReportBtn = document.getElementById("downloadReportBtn");
  const startNewBtn = document.getElementById("startNewBtn");
  const repeatQuestion = document.getElementById("speakAgainBtn");
  const feedbackLoading = document.getElementById("feedbackLoading");

  const steps = [
    document.getElementById("step1"),
    document.getElementById("step2"),
    document.getElementById("step3"),
    document.getElementById("step4"),
  ];
  const progressBar = document.getElementById("progressBar");

  function setActiveStep(stepIndex) {
    steps.forEach((step, index) => {
      step.classList.toggle("active", index === stepIndex);
      step.classList.toggle("completed", index < stepIndex);
    });

    progressBar.style.width = `${(stepIndex / (steps.length - 1)) * 100}%`;

    uploadSection.classList.add("hidden");
    processingSection.classList.add("hidden");
    interviewSection.classList.add("hidden");
    feedbackSection.classList.add("hidden");

    switch (stepIndex) {
      case 0: uploadSection.classList.remove("hidden"); break;
      case 1: processingSection.classList.remove("hidden"); break;
      case 2: interviewSection.classList.remove("hidden"); break;
      case 3:
        feedbackSection.classList.remove("hidden");
        feedbackSection.style.display = "block";
        break;
    }
  }

  async function handleResumeUpload(e) {
    e.preventDefault();
    const resumeFile = document.getElementById("resumeFile").files[0];
    if (!resumeFile) return alert("Please select a file.");

    setActiveStep(1);
    try {
      const formData = new FormData();
      formData.append("file", resumeFile);
      const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST", body: formData,
      });
      // ✅ Add this check
      if (!uploadRes.ok) {
        const errorText = await uploadRes.text(); // fallback to text to catch HTML
        throw new Error(`Upload failed: ${errorText}`);
      }
      const { filepath } = await uploadRes.json();

      const processRes = await fetch(`${API_BASE_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath }),
      });
      const processData = await processRes.json();

      const interviewRes = await fetch(`${API_BASE_URL}/interview/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ questions: processData.result.questions }),
      });
      const interviewData = await interviewRes.json();

      sessionId = interviewData.session_id;
      totalQuestions = interviewData.total_questions;
      currentQuestionIndex = 0;

      currentQuestion.textContent = interviewData.current_question.text;
      questionCounter.textContent = `Question 1/${totalQuestions}`;

      setActiveStep(2);
      playQuestionAudio(); // ONLY play here
    } catch (err) {
      alert("Upload failed: " + err.message);
      setActiveStep(0);
    }
  }

  async function submitTextAnswer() {
    if (!sessionId) return alert("Session not found.");
    const answer = answerText.value.trim();
    if (!answer) return alert("Please enter an answer.");

    submitAnswerBtn.disabled = true;
    submitAnswerBtn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Submitting...`;

    try {
      const res = await fetch(`${API_BASE_URL}/interview/submit_answer/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer }),
      });
      const data = await res.json();
      displayEvaluation(data.evaluation);
    } catch (err) {
      alert("Submission failed: " + err.message);
    } finally {
      submitAnswerBtn.disabled = false;
      submitAnswerBtn.innerHTML = `<i class="fas fa-paper-plane me-1"></i> Submit`;
    }
  }

  async function submitAudioAnswer(audioBlob) {
    if (!sessionId) return;

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.wav");

      submitAnswerBtn.disabled = true;
      startRecordingBtn.disabled = true;
      submitAnswerBtn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Processing...`;

      const res = await fetch(`${API_BASE_URL}/interview/audio_answer/${sessionId}`, {
        method: "POST", body: formData,
      });
      const data = await res.json();
      answerText.value = data.transcription;
      displayEvaluation(data.evaluation);
    } catch (err) {
      alert("Audio submit failed: " + err.message);
    } finally {
      submitAnswerBtn.disabled = false;
      startRecordingBtn.disabled = false;
      submitAnswerBtn.innerHTML = `<i class="fas fa-paper-plane me-1"></i> Submit`;
    }
  }

  function displayEvaluation(evaluation) {
    const evalData = JSON.parse(evaluation.evaluation);
    answerScore.textContent = evalData.score;
    answerCategory.textContent = evalData.category;
    answerFeedback.textContent = evalData.feedback;

    const categoryClass = {
      technical: "bg-info",
      behavioral: "bg-warning",
      role_specific: "bg-success",
    }[evalData.category.toLowerCase()] || "bg-secondary";

    answerCategory.className = `badge ${categoryClass}`;
    evaluationCard.classList.remove("hidden");

    nextQuestionBtn.disabled = false;
    if (evaluation.is_completed) {
      nextQuestionBtn.textContent = "Finish Interview";
      nextQuestionBtn.classList.replace("btn-primary", "btn-success");
      nextQuestionBtn.onclick = finalizeInterview;
    } else {
      nextQuestionBtn.textContent = "Next Question";
      nextQuestionBtn.classList.replace("btn-success", "btn-primary");
      nextQuestionBtn.onclick = loadNextQuestion;
    }
  }

  async function loadNextQuestion() {
    if (!sessionId) return;

    nextQuestionBtn.disabled = true;
    evaluationCard.classList.add("hidden");
    answerText.value = "";

    try {
      const res = await fetch(`${API_BASE_URL}/interview/next_question/${sessionId}`);
      const data = await res.json();
      if (data.error === "No more questions") return finalizeInterview();

      currentQuestionIndex++;
      currentQuestion.textContent = data.question;
      questionCounter.textContent = `Question ${currentQuestionIndex + 1}/${totalQuestions}`;
      playQuestionAudio();
    } catch (err) {
      alert("Failed to load next question.");
    } finally {
      nextQuestionBtn.disabled = false;
    }
  }

  async function finalizeInterview() {
    console.log("Finalizing interview...");
    setActiveStep(3);

    feedbackLoading.classList.remove("hidden");
    finalFeedbackContent.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE_URL}/interview/finalize/${sessionId}`, {
        method: "POST",
      });
      const data = await res.json();

      feedbackLoading.classList.add("hidden");

      if (!data.final_feedback) {
        finalFeedbackContent.innerHTML = "<p>No feedback available.</p>";
        return;
      }

      const lines = data.final_feedback.split("\n").filter(line => line.trim());
      let html = "";
      for (const line of lines) {
        if (line.startsWith("- ")) {
          html += `<li>${line.substring(2)}</li>`;
        } else if (line.includes(":")) {
          const [title, text] = line.split(":", 2);
          html += `<h6>${title.trim()}:</h6><p>${text.trim()}</p>`;
        } else {
          html += `<p>${line}</p>`;
        }
      }

      finalFeedbackContent.innerHTML = html;
      downloadReportBtn.href = `${API_BASE_URL}${data.report_url}`;
      downloadReportBtn.classList.remove("hidden");
      downloadReportBtn.style.display = "inline-block";
    } catch (err) {
      feedbackLoading.classList.add("hidden");
      finalFeedbackContent.innerHTML = "<p>Error generating feedback.</p>";
    }
  }

  function playQuestionAudio() {
    if (!sessionId) return;
    const audio = new Audio(`${API_BASE_URL}/speak_question/${sessionId}?t=${Date.now()}`);
    audio.play().catch(console.error);
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        submitAudioAnswer(audioBlob);
      };

      mediaRecorder.start();
      startRecordingBtn.disabled = true;
      stopRecordingBtn.disabled = false;
      recordingStatus.style.display = "inline";
    } catch (err) {
      alert("Mic access denied.");
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      startRecordingBtn.disabled = false;
      stopRecordingBtn.disabled = true;
      recordingStatus.style.display = "none";
    }
  }

  function resetInterview() {
    sessionId = null;
    currentQuestionIndex = 0;
    totalQuestions = 0;
    answerText.value = "";
    evaluationCard.classList.add("hidden");
    setActiveStep(0);
  }

  // 🔧 Only attach fixed listeners (no dupes)
  resumeUploadForm.addEventListener("submit", handleResumeUpload);
  submitAnswerBtn.addEventListener("click", submitTextAnswer);
  startRecordingBtn.addEventListener("click", startRecording);
  stopRecordingBtn.addEventListener("click", stopRecording);
  startNewBtn.addEventListener("click", resetInterview);
  repeatQuestion.addEventListener("click", playQuestionAudio);

  setActiveStep(0);
});