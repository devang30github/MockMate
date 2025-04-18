// src/App.jsx

import React, { useState } from "react";
import Header from "./components/Header";
import UploadResume from "./components/UploadResume";
import StartInterview from "./components/StartInterview"; // Import the StartInterview component
import Feedback from "./components/Feedback"; // Import the Feedback component
import "./App.css";

function App() {
  const [interviewResults, setInterviewResults] = useState(null);

  const handleInterviewCompletion = (results) => {
    setInterviewResults(results); // Store results after interview completion
  };

  return (
    <div className="App">
      <Header />

      <main>
        <section id="upload">
          <UploadResume /> {/* Include the UploadResume component */}
        </section>

        <section id="interview">
          <StartInterview onInterviewComplete={handleInterviewCompletion} /> {/* Pass callback */}
        </section>

        <section id="feedback">
          <Feedback interviewResults={interviewResults} /> {/* Pass interview results */}
        </section>

        <section id="contact">
          <h2>Contact</h2>
          {/* Contact information or form */}
        </section>
      </main>
    </div>
  );
}

export default App;
