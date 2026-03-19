/**
 * Auth Store
 * ==========
 *
 * Zustand store for managing authentication state with localStorage persistence.
 * Handles JWT tokens, login, register, logout, and token refresh.
 */

import { create } from "zustand";
import type { User } from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  isAuthenticated: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (data: {
    email: string;
    password: string;
    full_name: string;
    tenant_slug?: string;
  }) => Promise<{ message: string }>;
  logout: () => void;
  refreshAccessToken: () => Promise<boolean>;
  setAuth: (token: string, refreshToken: string, user: User) => void;
}

// Load persisted auth from localStorage
function loadPersistedAuth(): {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
} {
  try {
    const token = localStorage.getItem("auth_token");
    const refreshToken = localStorage.getItem("auth_refresh_token");
    const userStr = localStorage.getItem("auth_user");
    const user = userStr ? JSON.parse(userStr) : null;
    return { token, refreshToken, user };
  } catch {
    return { token: null, refreshToken: null, user: null };
  }
}

function persistAuth(token: string | null, refreshToken: string | null, user: User | null) {
  if (token) {
    localStorage.setItem("auth_token", token);
  } else {
    localStorage.removeItem("auth_token");
  }
  if (refreshToken) {
    localStorage.setItem("auth_refresh_token", refreshToken);
  } else {
    localStorage.removeItem("auth_refresh_token");
  }
  if (user) {
    localStorage.setItem("auth_user", JSON.stringify(user));
  } else {
    localStorage.removeItem("auth_user");
  }
}

const initial = loadPersistedAuth();

export const useAuthStore = create<AuthState>((set, get) => ({
  token: initial.token,
  refreshToken: initial.refreshToken,
  user: initial.user,
  isAuthenticated: !!initial.token && !!initial.user,

  setAuth: (token, refreshToken, user) => {
    persistAuth(token, refreshToken, user);
    set({ token, refreshToken, user, isAuthenticated: true });
  },

  login: async (email, password) => {
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Login failed" }));
      throw new Error(err.detail || `Login error: ${res.status}`);
    }

    const data = await res.json();
    const { access_token, refresh_token, user } = data;
    persistAuth(access_token, refresh_token, user);
    set({
      token: access_token,
      refreshToken: refresh_token,
      user,
      isAuthenticated: true,
    });
  },

  register: async (data) => {
    const res = await fetch(`${BASE_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Registration failed" }));
      throw new Error(err.detail || `Register error: ${res.status}`);
    }

    return { message: "Account created. Please wait for admin approval." };
  },

  logout: () => {
    persistAuth(null, null, null);
    set({
      token: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,
    });
  },

  refreshAccessToken: async () => {
    const { refreshToken } = get();
    if (!refreshToken) return false;

    try {
      const res = await fetch(`${BASE_URL}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!res.ok) {
        // Refresh failed — logout
        get().logout();
        return false;
      }

      const data = await res.json();
      const newToken = data.access_token;
      localStorage.setItem("auth_token", newToken);
      set({ token: newToken });
      return true;
    } catch {
      get().logout();
      return false;
    }
  },
}));
