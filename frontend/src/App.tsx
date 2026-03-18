import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { useThemeStore } from "@/stores/useThemeStore";
import { AppShell } from "@/components/layout/AppShell";
import { KnowledgeBasesPage } from "@/pages/KnowledgeBasesPage";
import { WorkspacePage } from "@/pages/WorkspacePage";
import { FilesPage } from "@/pages/FilesPage";
import { WorkersPage } from "@/pages/WorkersPage";

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
      <Route element={<AppShell />}>
        <Route path="/" element={<KnowledgeBasesPage />} />
        <Route path="/knowledge-bases/:workspaceId" element={<WorkspacePage />} />
        <Route path="/knowledge-bases/:workspaceId/files" element={<FilesPage />} />
        <Route path="/files" element={<FilesPage />} />
        <Route path="/workers" element={<WorkersPage />} />
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
