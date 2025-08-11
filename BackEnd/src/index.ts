import express from "express";
import cors from "cors";
const app = express();
const port = Number(process.env.PORT) || 3000;

app.use(express.json());
app.use(cors());

app.get("/", (_req, res) => {
  res.send({ message: "Hello, world!" });
});

app.post("/api/chat", async (req, res) => {
  const { query, history } = req.body;
  console.log(query);
  const response = await fetch("http://localhost:8000/it-assistant-v2", {
    method: "POST",
    body: JSON.stringify({ query: query, history: history }),
    headers: {
      "Content-Type": "application/json",
    },
  });
  const data = await response.json();
  res.send(data);
});

app.post("/tools/passwordReset", (req, res) => {
  const { query } = req.body;
  res.send({ message: "Password reset API has been called" });
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
