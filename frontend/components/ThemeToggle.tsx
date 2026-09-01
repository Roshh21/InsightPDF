"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";

export default function ThemeToggle({ collapsed }: { collapsed?: boolean }) {
  const { resolvedTheme, setTheme } = useTheme();
  // Avoid a hydration mismatch: next-themes only knows the real resolved
  // theme after mount (it reads localStorage/system preference client-side).
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const isDark = mounted && resolvedTheme === "dark";

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="flex w-full items-center gap-2.5 rounded-[5px] px-2.5 py-2 text-[13.5px] font-medium text-ink-soft transition-colors hover:bg-paper-dim hover:text-ink"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {mounted && isDark ? (
        <Moon size={16} strokeWidth={2} className="text-ink-faint" />
      ) : (
        <Sun size={16} strokeWidth={2} className="text-ink-faint" />
      )}
      {!collapsed && <span>{mounted ? (isDark ? "Dark mode" : "Light mode") : "Theme"}</span>}
    </button>
  );
}
