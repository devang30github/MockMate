// src/components/Feedback.jsx

import React, { useState, useEffect } from "react";
import "./Feedback.css"; // Import the CSS for styling

const Feedback = ({ interviewResults }) => {
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    // Simulating an API call or processing the results
    if (interviewResults) {
      generateFeedback(interviewResults);
    }
  }, [interviewResults]);

  const generateFeedback = (results) => {
    // Mock function to simulate feedback generation based on results
    let averageScore = results.reduce((acc, result) => acc + result.evaluation.score, 0) / results.length;
    let feedbackText = `The candidate's overall performance was rated an average of ${averageScore.toFixed(1)}.`;

    // Additional feedback logic can be added here based on the results

    setFeedback({
      summary: `The candidate's performance was good, with scores ranging from ${Math.min(...results.map(r => r.evaluation.score))} to ${Math.max(...results.map(r => r.evaluation.score))}.`,
      strengths: "Strengths include good technical understanding, and behavioral responses showing growth potential.",
      weaknesses: "Weaknesses include lack of depth in technical details and failure to provide specific examples in behavioral questions.",
      suggestions: "Suggestions include improving technical knowledge, practicing behavioral interview responses, and reviewing key concepts related to the role.",
    });
  };

  return (
    <div className="feedback-container">
      <h2>Interview Feedback</h2>

      {feedback ? (
        <div className="feedback-content">
          <h3>Summary of Performance</h3>
          <p>{feedback.summary}</p>

          <h3>Strengths</h3>
          <p>{feedback.strengths}</p>

          <h3>Weaknesses</h3>
          <p>{feedback.weaknesses}</p>

          <h3>Suggestions for Improvement</h3>
          <p>{feedback.suggestions}</p>
        </div>
      ) : (
        <p>Loading feedback...</p>
      )}
    </div>
  );
};

export default Feedback;
