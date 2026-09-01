/** @type {import('tailwindcss').Config} */
function themeColor(name) {
  return `rgb(var(--color-${name}) / <alpha-value>)`;
}

module.exports = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: themeColor("paper"),
          soft: themeColor("paper-soft"),
          dim: themeColor("paper-dim"),
        },
        ink: {
          DEFAULT: themeColor("ink"),
          soft: themeColor("ink-soft"),
          faint: themeColor("ink-faint"),
        },
        cobalt: {
          DEFAULT: themeColor("cobalt"),
          dark: themeColor("cobalt-dark"),
          soft: themeColor("cobalt-soft"),
        },
        amber: {
          DEFAULT: themeColor("amber"),
          dark: themeColor("amber-dark"),
          soft: themeColor("amber-soft"),
        },
        line: {
          DEFAULT: themeColor("line"),
          soft: themeColor("line-soft"),
        },
        success: { DEFAULT: themeColor("success"), soft: themeColor("success-soft") },
        danger: { DEFAULT: themeColor("danger"), soft: themeColor("danger-soft") },
      },
      fontFamily: {
        serif: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "6px",
        md: "8px",
        lg: "12px",
      },
      boxShadow: {
        card: "0 1px 2px rgb(0 0 0 / 0.04), 0 1px 8px rgb(0 0 0 / 0.04)",
        pop: "0 4px 16px rgb(0 0 0 / 0.12)",
      },
    },
  },
  plugins: [],
};
