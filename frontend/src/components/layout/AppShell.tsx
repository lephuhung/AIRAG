import { useState, useEffect, useCallback, useRef } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const STORAGE_KEY = "sidebar-collapsed";
const AUTO_COLLAPSE_WIDTH = 1280;

export function AppShell() {
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored !== null) return stored === "true";
    return window.innerWidth < AUTO_COLLAPSE_WIDTH;
  });

  const lastWidth = useRef(window.innerWidth);

  const toggleSidebar = useCallback(() => {
    setIsCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(STORAGE_KEY, String(next));
      return next;
    });
  }, []);

  useEffect(() => {
    const check = () => {
      const width = window.innerWidth;
      const wasWide = lastWidth.current >= AUTO_COLLAPSE_WIDTH;
      const isNarrow = width < AUTO_COLLAPSE_WIDTH;

      if (isNarrow && wasWide) {
        setIsCollapsed(true);
      } else if (!isNarrow && wasWide === false) {
        const stored = localStorage.getItem(STORAGE_KEY);
        setIsCollapsed(stored === "true");
      }
      
      lastWidth.current = width;
    };

    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  return (
    <div className="h-screen flex overflow-hidden relative">
      <div className="w-14 flex-shrink-0" aria-hidden="true" />
      <Sidebar collapsed={isCollapsed} onToggle={toggleSidebar} />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
