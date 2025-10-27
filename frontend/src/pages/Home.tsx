import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Home.css';

const Home: React.FC = () => {
  return (
    <div className="home-container">
      {/* 背景装饰 */}
      <div className="home-background">
        <div className="bg-pattern"></div>
      </div>

      {/* 主要内容 */}
      <div className="home-content">
        {/* 标题区域 */}
        <div className="hero-section">
          <h1 className="hero-title">
            GPT-SoVITS
            <span className="hero-subtitle">语音合成系统</span>
          </h1>
          <p className="hero-description">
            基于先进的人工智能技术，为您提供自然流畅的语音合成体验
          </p>
        </div>

        {/* 功能卡片 */}
        <div className="features-grid">
          <Link to="/tts-chat" className="feature-card">
            <div className="card-icon">🎤</div>
            <h3>语音对话</h3>
            <p>与AI进行智能语音对话，体验自然的交互</p>
          </Link>

          <div className="feature-card coming-soon">
            <div className="card-icon">🔊</div>
            <h3>语音合成</h3>
            <p>即将推出更多语音合成功能</p>
            <span className="coming-soon-badge">敬请期待</span>
          </div>

          <div className="feature-card coming-soon">
            <div className="card-icon">🎵</div>
            <h3>音频处理</h3>
            <p>即将推出音频编辑和处理工具</p>
            <span className="coming-soon-badge">敬请期待</span>
          </div>

          <div className="feature-card coming-soon">
            <div className="card-icon">📊</div>
            <h3>数据分析</h3>
            <p>即将推出语音数据分析功能</p>
            <span className="coming-soon-badge">敬请期待</span>
          </div>
        </div>

        {/* 开始使用按钮 */}
        <div className="cta-section">
          <Link to="/tts-chat" className="cta-button">
            开始体验语音对话
            <span className="arrow">→</span>
          </Link>
        </div>
      </div>

      {/* 页脚 */}
      <footer className="home-footer">
        <p>&copy; 2024 GPT-SoVITS 语音合成系统. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Home;
