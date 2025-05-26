// Install dependencies:
// - npx create-react-app legal-advisor-ui
// - cd legal-advisor-ui
// - npm install axios
// */

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const handleSubmit = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/ask', { query });
      setAnswer(res.data.answer);
    } catch (error) {
      setAnswer('Error retrieving answer.');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Legal Advisor AI Chatbot</h1>
      <textarea
        rows="4"
        cols="60"
        placeholder="Ask your legal question..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      ></textarea>
      <br />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Thinking...' : 'Submit'}
      </button>
      <div className="answer">
        <h3>Answer:</h3>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;

// ----------------------------
// 3. Run Instructions
// ----------------------------
// Terminal 1: Start backend API
// uvicorn backend:app --reload

// Terminal 2: Start frontend
// npm start (in legal-advisor-ui directory)

// Make sure the backend is accessible on http://localhost:8000
