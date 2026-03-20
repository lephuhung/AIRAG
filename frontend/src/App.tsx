import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { useThemeStore } from "@/stores/useThemeStore";
import { AppShell } from "@/components/layout/AppShell";
import { ProtectedRoute } from "@/components/layout/ProtectedRoute";
import { KnowledgeBasesPage } from "@/pages/KnowledgeBasesPage";
import { WorkspacePage } from "@/pages/WorkspacePage";
import { FilesPage } from "@/pages/FilesPage";
import { WorkersPage } from "@/pages/WorkersPage";
import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { TenantManagePage } from "@/pages/TenantManagePage";
import { AdminUsersPage } from "@/pages/AdminUsersPage";
import { AdminTenantsPage } from "@/pages/AdminTenantsPage";
import { AdminDocumentTypesPage } from "@/pages/AdminDocumentTypesPage";
import { ChatPage } from "@/pages/ChatPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function AppRoutes() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Protected routes */}
      <Route
        element={
          <ProtectedRoute>
            <AppShell />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<KnowledgeBasesPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:sessionId" element={<ChatPage />} />
        <Route path="/knowledge-bases/:workspaceId" element={<WorkspacePage />} />
        <Route path="/knowledge-bases/:workspaceId/files" element={<FilesPage />} />
        <Route path="/files" element={<FilesPage />} />
        <Route path="/workers" element={<WorkersPage />} />
        <Route path="/admin/users" element={<AdminUsersPage />} />
        <Route path="/admin/tenants" element={<AdminTenantsPage />} />
        <Route path="/admin/document-types" element={<AdminDocumentTypesPage />} />
        <Route path="/tenants/:tenantId" element={<TenantManagePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  const theme = useThemeStore((s) => s.theme);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
      <Toaster
        theme={theme}
        position="bottom-right"
        richColors
        toastOptions={{
          duration: 4000,
          className: "text-sm",
        }}
      />
    </QueryClientProvider>
  );
}

export default App;
