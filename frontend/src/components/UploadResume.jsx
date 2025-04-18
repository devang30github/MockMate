// src/components/UploadResume.jsx

import React, { useState } from "react";
import "./UploadResume.css"; // Import the CSS for styling

const UploadResume = () => {
  const [file, setFile] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  // Handle file submission (you can connect this to an API to store the resume)
  const handleSubmit = (event) => {
    event.preventDefault();
    if (file) {
      alert(`File uploaded: ${file.name}`);
      // Implement the logic to send the file to the server for further processing
    } else {
      alert("Please select a file to upload.");
    }
  };

  return (
    <div className="upload-resume">
      <h2>Upload Your Resume</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input">
          <label htmlFor="resume-upload" className="upload-label">
            Choose a PDF file
          </label>
          <input
            type="file"
            id="resume-upload"
            accept=".pdf"
            onChange={handleFileChange}
            className="file-input-field"
          />
        </div>

        {file && (
          <div className="file-info">
            <p>File selected: <strong>{file.name}</strong></p>
          </div>
        )}

        <button type="submit" className="upload-button">
          Upload Resume
        </button>
      </form>
    </div>
  );
};

export default UploadResume;
