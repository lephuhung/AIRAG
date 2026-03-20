import { useEffect } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    // Buffer of 60 seconds
    return payload.exp * 1000 < Date.now() + 60000;
  } catch {
    return true;
  }
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const token = useAuthStore((s) => s.token);
  const refreshAccessToken = useAuthStore((s) => s.refreshAccessToken);
  const location = useLocation();

  // Actively monitor token expiration
  useEffect(() => {
    if (!isAuthenticated || !token) return;

    // Check immediately on mount or token change
    if (isTokenExpired(token)) {
      refreshAccessToken();
    }

    // Periodic check every 30 seconds
    const interval = setInterval(() => {
      const currentToken = useAuthStore.getState().token;
      if (currentToken && isTokenExpired(currentToken)) {
        refreshAccessToken();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [isAuthenticated, token, refreshAccessToken]);

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
