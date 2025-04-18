// src/components/Header.jsx

import React from "react";
import "./Header.css"; // Import the CSS for styling

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <h1>MockMate</h1>
        <p>Your AI-Powered Interview Buddy</p>
      </div>
      <nav className="nav-links">
        <ul>
          <li><a href="#upload">Upload Resume</a></li>
          <li><a href="#interview">Start Interview</a></li>
          <li><a href="#feedback">Feedback</a></li>
          <li><a href="#contact">Contact</a></li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;
