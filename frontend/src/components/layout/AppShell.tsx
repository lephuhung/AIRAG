import { useState, useEffect, useCallback, useRef } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "sidebar-collapsed";
const AUTO_COLLAPSE_WIDTH = 1280;

export function AppShell() {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored !== null) return stored === "true";
    return window.innerWidth < AUTO_COLLAPSE_WIDTH;
  });

  const [isNarrow, setIsNarrow] = useState(window.innerWidth < AUTO_COLLAPSE_WIDTH);
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
      const currentlyNarrow = width < AUTO_COLLAPSE_WIDTH;
      setIsNarrow(currentlyNarrow);
      const wasWide = lastWidth.current >= AUTO_COLLAPSE_WIDTH;

      if (currentlyNarrow && wasWide) {
        setIsCollapsed(true);
      } else if (!currentlyNarrow && wasWide === false) {
        const stored = localStorage.getItem(STORAGE_KEY);
        setIsCollapsed(stored === "true");
      }
      
      lastWidth.current = width;
    };

    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Auto-collapse on navigation if screen is narrow
  useEffect(() => {
    if (isNarrow && !isCollapsed) {
      setIsCollapsed(true);
    }
  }, [location.pathname, isNarrow, isCollapsed]);

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--sidebar-width",
      isCollapsed ? "56px" : "240px",
    );
  }, [isCollapsed]);

  return (
    <div className="h-screen flex overflow-hidden relative bg-background">
      {/* Backdrop for mobile/narrow screens when sidebar is expanded */}
      <AnimatePresence>
        {!isCollapsed && isNarrow && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsCollapsed(true)}
            className="fixed inset-0 bg-black/40 z-40 backdrop-blur-[2px] cursor-pointer"
            aria-hidden="true"
          />
        )}
      </AnimatePresence>

      {/* Layout Spacer for Desktop: ensures content doesn't go under the persistent sidebar */}
      {!isNarrow && (
        <div 
          className={cn("flex-shrink-0 transition-all duration-300", isCollapsed ? "w-14" : "w-60")} 
          aria-hidden="true" 
        />
      )}

      <Sidebar collapsed={isCollapsed} onToggle={toggleSidebar} isNarrow={isNarrow} />
      
      <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-background">
        <TopBar onToggle={toggleSidebar} isNarrow={isNarrow} />
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
