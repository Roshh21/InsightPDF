"use client";

import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import { useEffect, useState } from "react";
import {
  BarChart3,
  BookOpenText,
  Cpu,
  FolderKanban,
  Globe2,
  Menu,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Rows3,
  Settings,
  SquareStack,
  X,
} from "lucide-react";
import { api } from "@/lib/api";
import type { Workspace } from "@/lib/types";
import ThemeToggle from "./ThemeToggle";

const SIDEBAR_STORAGE_KEY = "insightpdf-sidebar-collapsed";

function Logo({ showLabel = true }: { showLabel?: boolean }) {
  return (
    <span className="flex items-center gap-2 overflow-hidden">
      <span className="flex h-6 w-5 shrink-0 items-center justify-center rounded-[3px] border-l-[3px] border-amber bg-amber-soft">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-dark" />
      </span>
      {showLabel && (
        <span className="truncate font-serif text-[17px] font-semibold tracking-tight text-ink">
          InsightPDF
        </span>
      )}
    </span>
  );
}

function NavItem({
  href,
  icon: Icon,
  label,
  active,
  collapsed,
  onNavigate,
}: {
  href: string;
  icon: any;
  label: string;
  active: boolean;
  collapsed?: boolean;
  onNavigate?: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onNavigate}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      className={`group flex items-center gap-2.5 rounded-[5px] py-2 text-[13.5px] font-medium transition-colors ${
        collapsed ? "justify-center px-2" : "px-2.5"
      } ${active ? "bg-cobalt-soft text-cobalt-dark" : "text-ink-soft hover:bg-paper-dim hover:text-ink"}`}
    >
      <Icon
        size={16}
        strokeWidth={2}
        className={`shrink-0 ${active ? "text-cobalt" : "text-ink-faint group-hover:text-ink-soft"}`}
      />
      {!collapsed && <span className="truncate">{label}</span>}
    </Link>
  );
}

function SidebarSections({
  pathname,
  workspaceId,
  workspace,
  collapsed,
  onNavigate,
}: {
  pathname: string;
  workspaceId?: string;
  workspace: Workspace | null;
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  return (
    <>
      {workspaceId && (
        <div className="mb-4">
          {!collapsed && (
            <>
              <div className="mb-1.5 px-2.5 text-[10.5px] font-semibold uppercase tracking-wider text-ink-faint">
                Workspace
              </div>
              <div
                className="mb-2 truncate px-2.5 font-serif text-[14px] font-semibold text-ink"
                title={workspace?.name}
              >
                {workspace?.name || "Loading…"}
              </div>
            </>
          )}
          <nav className="flex flex-col gap-0.5">
            <NavItem
              href={`/workspaces/${workspaceId}`}
              icon={Rows3}
              label="Documents"
              active={pathname === `/workspaces/${workspaceId}`}
              collapsed={collapsed}
              onNavigate={onNavigate}
            />
            <NavItem
              href={`/workspaces/${workspaceId}/chat`}
              icon={MessageSquare}
              label="Chat"
              active={pathname.startsWith(`/workspaces/${workspaceId}/chat`)}
              collapsed={collapsed}
              onNavigate={onNavigate}
            />
            <NavItem
              href={`/workspaces/${workspaceId}/compare`}
              icon={SquareStack}
              label="Compare"
              active={pathname.startsWith(`/workspaces/${workspaceId}/compare`)}
              collapsed={collapsed}
              onNavigate={onNavigate}
            />
            <NavItem
              href={`/workspaces/${workspaceId}/research`}
              icon={Globe2}
              label="Web Research"
              active={pathname.startsWith(`/workspaces/${workspaceId}/research`)}
              collapsed={collapsed}
              onNavigate={onNavigate}
            />
          </nav>
        </div>
      )}

      <div>
        {!collapsed && (
          <div className="mb-1.5 px-2.5 text-[10.5px] font-semibold uppercase tracking-wider text-ink-faint">
            Platform
          </div>
        )}
        <nav className="flex flex-col gap-0.5">
          <NavItem href="/" icon={FolderKanban} label="Workspaces" active={pathname === "/"} collapsed={collapsed} onNavigate={onNavigate} />
          <NavItem href="/models" icon={Cpu} label="Model Status" active={pathname === "/models"} collapsed={collapsed} onNavigate={onNavigate} />
          <NavItem
            href="/evaluation"
            icon={BarChart3}
            label="Evaluation"
            active={pathname === "/evaluation"}
            collapsed={collapsed}
            onNavigate={onNavigate}
          />
        </nav>
      </div>

      <div className="mt-auto pt-3">
        <nav className="flex flex-col gap-0.5 border-t border-line pt-2">
          <ThemeToggle collapsed={collapsed} />
          <NavItem href="/settings" icon={Settings} label="Settings" active={pathname === "/settings"} collapsed={collapsed} onNavigate={onNavigate} />
        </nav>
        <div
          className={`mt-3 flex items-center gap-1.5 text-[11px] text-ink-faint ${collapsed ? "justify-center px-0" : "px-2.5"}`}
          title={collapsed ? "Agentic RAG Platform" : undefined}
        >
          <BookOpenText size={13} className="shrink-0" />
          {!collapsed && "Agentic RAG Platform"}
        </div>
      </div>
    </>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || "/";
  const params = useParams();
  const workspaceId = typeof params?.workspaceId === "string" ? params.workspaceId : undefined;
  const [workspace, setWorkspace] = useState<Workspace | null>(null);

  // Desktop/tablet: sidebar collapses to an icon-only rail. Persisted so the
  // choice survives a refresh, but we only trust it after mount to avoid a
  // server/client markup mismatch.
  const [collapsed, setCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Mobile (<md): sidebar becomes an off-canvas drawer, closed by default.
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMounted(true);
    try {
      if (window.localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1") {
        setCollapsed(true);
      }
    } catch {
      // localStorage unavailable (private browsing, etc.) -- default to expanded.
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;
    try {
      window.localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      // Ignore -- persistence is a nice-to-have, not a requirement.
    }
  }, [collapsed, mounted]);

  // Close the mobile drawer on navigation.
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Close the mobile drawer if the viewport grows past the mobile breakpoint
  // (e.g. rotating a tablet, or dragging a browser window wider).
  useEffect(() => {
    function handleResize() {
      if (window.innerWidth >= 768) {
        setMobileOpen(false);
      }
    }
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Prevent the page behind the drawer from scrolling while it's open.
  useEffect(() => {
    if (!mobileOpen) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [mobileOpen]);

  useEffect(() => {
    if (!workspaceId) {
      setWorkspace(null);
      return;
    }
    let cancelled = false;
    api
      .getWorkspace(workspaceId)
      .then((w) => !cancelled && setWorkspace(w))
      .catch(() => !cancelled && setWorkspace(null));
    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  const isAuth = pathname === "/__never__"; // placeholder, no auth screens currently

  if (isAuth) return <>{children}</>;

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      {/* Mobile top bar: logo + hamburger. Hidden from md upward, where the
          sidebar itself is always in view. */}
      <div className="flex items-center justify-between border-b border-line bg-paper-soft/60 px-4 py-3 md:hidden">
        <Link href="/">
          <Logo />
        </Link>
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          aria-label="Open menu"
          aria-expanded={mobileOpen}
          className="flex h-8 w-8 items-center justify-center rounded-[5px] text-ink-soft transition-colors hover:bg-paper-dim hover:text-ink"
        >
          <Menu size={18} />
        </button>
      </div>

      {/* Mobile off-canvas drawer + backdrop. Always mounted (for a smooth
          slide transition) but inert and invisible until opened, and
          removed from layout entirely from md upward. */}
      <div className={`fixed inset-0 z-40 md:hidden ${mobileOpen ? "" : "pointer-events-none"}`}>
        <div
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
          className={`absolute inset-0 bg-ink/40 transition-opacity duration-200 ${
            mobileOpen ? "opacity-100" : "opacity-0"
          }`}
        />
        <aside
          role="dialog"
          aria-modal="true"
          aria-label="Navigation menu"
          className={`relative flex h-full w-[248px] max-w-[80vw] flex-col overflow-y-auto border-r border-line bg-paper-soft px-3 py-4 shadow-pop transition-transform duration-200 ease-in-out ${
            mobileOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="mb-5 flex items-center justify-between px-1.5">
            <Link href="/" onClick={() => setMobileOpen(false)}>
              <Logo />
            </Link>
            <button
              type="button"
              onClick={() => setMobileOpen(false)}
              aria-label="Close menu"
              className="flex h-7 w-7 items-center justify-center rounded-[5px] text-ink-faint transition-colors hover:bg-paper-dim hover:text-ink"
            >
              <X size={16} />
            </button>
          </div>
          <SidebarSections
            pathname={pathname}
            workspaceId={workspaceId}
            workspace={workspace}
            collapsed={false}
            onNavigate={() => setMobileOpen(false)}
          />
        </aside>
      </div>

      {/* Desktop / tablet sidebar: static, collapsible to an icon rail. */}
      <aside
        className={`sticky top-0 hidden h-screen shrink-0 flex-col overflow-y-auto border-r border-line bg-paper-soft/60 py-4 transition-[width] duration-200 ease-in-out md:flex ${
          collapsed ? "w-[68px] px-2" : "w-[228px] px-3"
        }`}
      >
        <div className={`mb-5 flex items-center ${collapsed ? "flex-col gap-2 px-0" : "justify-between px-1.5"}`}>
          <Link href="/">
            <Logo showLabel={!collapsed} />
          </Link>
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[4px] text-ink-faint transition-colors hover:bg-paper-dim hover:text-ink"
          >
            {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
          </button>
        </div>
        <SidebarSections pathname={pathname} workspaceId={workspaceId} workspace={workspace} collapsed={collapsed} />
      </aside>

      <main className="min-w-0 flex-1">{children}</main>
    </div>
  );
}
