import { createBrowserRouter, RouterProvider } from "react-router";
import ChatPage from "@/pages/chatPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <ChatPage />,
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
