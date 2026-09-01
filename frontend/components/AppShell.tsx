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
  LayoutGrid,
  MessageSquare,
  Rows3,
  Settings,
  SquareStack,
} from "lucide-react";
import { api } from "@/lib/api";
import type { Workspace } from "@/lib/types";
import ThemeToggle from "./ThemeToggle";

function NavItem({
  href,
  icon: Icon,
  label,
  active,
}: {
  href: string;
  icon: any;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={`group flex items-center gap-2.5 rounded-[5px] px-2.5 py-2 text-[13.5px] font-medium transition-colors ${
        active ? "bg-cobalt-soft text-cobalt-dark" : "text-ink-soft hover:bg-paper-dim hover:text-ink"
      }`}
    >
      <Icon size={16} strokeWidth={2} className={active ? "text-cobalt" : "text-ink-faint group-hover:text-ink-soft"} />
      {label}
    </Link>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || "/";
  const params = useParams();
  const workspaceId = typeof params?.workspaceId === "string" ? params.workspaceId : undefined;
  const [workspace, setWorkspace] = useState<Workspace | null>(null);

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
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-[228px] shrink-0 flex-col border-r border-line bg-paper-soft/60 px-3 py-4">
        <Link href="/" className="mb-5 flex items-center gap-2 px-1.5">
          <span className="flex h-6 w-5 items-center justify-center rounded-[3px] border-l-[3px] border-amber bg-amber-soft">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-dark" />
          </span>
          <span className="font-serif text-[17px] font-semibold tracking-tight text-ink">InsightPDF</span>
        </Link>

        {workspaceId && (
          <div className="mb-4">
            <div className="mb-1.5 px-2.5 text-[10.5px] font-semibold uppercase tracking-wider text-ink-faint">
              Workspace
            </div>
            <div className="mb-2 truncate px-2.5 font-serif text-[14px] font-semibold text-ink" title={workspace?.name}>
              {workspace?.name || "Loading…"}
            </div>
            <nav className="flex flex-col gap-0.5">
              <NavItem
                href={`/workspaces/${workspaceId}`}
                icon={Rows3}
                label="Documents"
                active={pathname === `/workspaces/${workspaceId}`}
              />
              <NavItem
                href={`/workspaces/${workspaceId}/chat`}
                icon={MessageSquare}
                label="Chat"
                active={pathname.startsWith(`/workspaces/${workspaceId}/chat`)}
              />
              <NavItem
                href={`/workspaces/${workspaceId}/compare`}
                icon={SquareStack}
                label="Compare"
                active={pathname.startsWith(`/workspaces/${workspaceId}/compare`)}
              />
              <NavItem
                href={`/workspaces/${workspaceId}/research`}
                icon={Globe2}
                label="Web Research"
                active={pathname.startsWith(`/workspaces/${workspaceId}/research`)}
              />
            </nav>
          </div>
        )}

        <div>
          <div className="mb-1.5 px-2.5 text-[10.5px] font-semibold uppercase tracking-wider text-ink-faint">
            Platform
          </div>
          <nav className="flex flex-col gap-0.5">
            <NavItem href="/" icon={FolderKanban} label="Workspaces" active={pathname === "/"} />
            <NavItem href="/models" icon={Cpu} label="Model Status" active={pathname === "/models"} />
            <NavItem
              href="/evaluation"
              icon={BarChart3}
              label="Evaluation"
              active={pathname === "/evaluation"}
            />
          </nav>
        </div>

        <div className="mt-auto pt-3">
          <nav className="flex flex-col gap-0.5 border-t border-line pt-2">
            <ThemeToggle />
            <NavItem href="/settings" icon={Settings} label="Settings" active={pathname === "/settings"} />
          </nav>
          <div className="mt-3 flex items-center gap-1.5 px-2.5 text-[11px] text-ink-faint">
            <BookOpenText size={13} />
            Agentic RAG Platform
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1">{children}</main>
    </div>
  );
}
