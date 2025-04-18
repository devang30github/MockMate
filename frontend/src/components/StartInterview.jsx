// src/components/StartInterview.jsx

import React, { useState } from "react";
import "./StartInterview.css"; // Import the CSS for styling

const StartInterview = () => {
  const [interviewStarted, setInterviewStarted] = useState(false);

  const handleStartInterview = () => {
    // This is where the logic for starting the interview would go
    setInterviewStarted(true);
  };

  return (
    <div className="start-interview">
      <h2>Start Your Mock Interview</h2>

      {!interviewStarted ? (
        <div className="start-button-container">
          <button onClick={handleStartInterview} className="start-button">
            Start Interview
          </button>
        </div>
      ) : (
        <div className="interview-in-progress">
          <p>The interview is in progress...</p>
          {/* You can add more dynamic content here */}
        </div>
      )}
    </div>
  );
};

export default StartInterview;
