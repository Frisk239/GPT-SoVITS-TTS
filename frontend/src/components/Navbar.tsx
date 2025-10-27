import React, { useState } from 'react';
import '../styles/Navbar.css';

const Navbar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Logo */}
        <div className="navbar-logo">
          <img
            src="/static/minpaixinyu-static/image/logo.png"
            alt="民派新语"
            className="logo-img"
          />
        </div>

        {/* 汉堡菜单按钮 - 只在移动端显示 */}
        <button
          className={`hamburger-menu ${isMenuOpen ? 'open' : ''}`}
          onClick={toggleMenu}
          aria-label="切换菜单"
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        {/* 导航链接 - 装饰性，无实际链接 */}
        <div className={`navbar-links ${isMenuOpen ? 'open' : ''}`}>
          <span className="nav-link">文化云游</span>
          <span className="nav-link">AI对话</span>
          <span className="nav-link">有声读物</span>
          <span className="nav-link">线上桌游</span>
          <span className="nav-link">互动答题</span>
          <span className="nav-link">个人中心</span>
        </div>
      </div>

      {/* 移动端菜单遮罩 */}
      {isMenuOpen && <div className="menu-overlay" onClick={closeMenu}></div>}
    </nav>
  );
};

export default Navbar;
