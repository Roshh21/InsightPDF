"use client";

import React from "react";
import Link from "next/link";

const NotFound = () => {
  return (
    <div className="card card-soft">
      <h1 style={{ marginTop: 0 }}>404 – Page not found</h1>

      <p style={{ color: "var(--text-muted)" }}>
        Looks like this page wandered off into the bookshelf.
      </p>

      <Link
        href="/"
        className="btn btn-ghost"
        style={{ textDecoration: "none" }}
      >
        Back to home
      </Link>
    </div>
  );
};

export default NotFound;
