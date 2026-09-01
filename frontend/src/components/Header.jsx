import React from "react";
import { Link, useLocation } from "react-router-dom";

const Header = () => {
  const location = useLocation();

  return (
    <header className="header-root">
      <div className="header-inner">
        <Link to="/" className="brand">
          <div className="brand-mark">IP</div>
          <div className="brand-text">
            <span className="brand-title">InsightPDF</span>
            <span className="brand-subtitle">
              Intelligent agentic document analyzer
            </span>
          </div>
        </Link>

        <nav className="nav-links">
          <Link
            to="/"
            className={
              location.pathname === "/" ? "nav-link nav-link-active" : "nav-link"
            }
          >
            Home
          </Link>
          <Link
            to="/analyzer"
            className={
              location.pathname === "/analyzer"
                ? "nav-link nav-link-active"
                : "nav-link"
            }
          >
            Analyzer
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;