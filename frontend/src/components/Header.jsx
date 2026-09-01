"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const Header = () => {
  const pathname = usePathname();

  return (
    <header className="header-root">
      <div className="header-inner">
        <Link href="/" className="brand">
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
            href="/"
            className={
              pathname === "/"
                ? "nav-link nav-link-active"
                : "nav-link"
            }
          >
            Home
          </Link>

          <Link
            href="/Analyzer"
            className={
              pathname === "/Analyzer"
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