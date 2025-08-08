import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ThemeProvider } from "@/components/themeProvider";
import { IconContext } from "react-icons";
import { SidebarProvider } from "@/components/ui/sidebar";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider defaultTheme='dark' storageKey='vite-ui-theme'>
      <IconContext.Provider value={{ color: "--foreground", className: "" }}>
        <SidebarProvider defaultOpen={true}>
          <App />
        </SidebarProvider>
      </IconContext.Provider>
    </ThemeProvider>
  </StrictMode>
);
