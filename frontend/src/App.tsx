import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import TTSChat from './pages/TTSChat';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/tts-chat" element={<TTSChat />} />
          <Route path="/about" element={<div style={{ padding: '100px', textAlign: 'center' }}>关于我们页面 - 敬请期待</div>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
