import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 120000,
});

export async function analyseProblem(question, product) {
  const response = await api.post("/analyse", {
    question,
    product,
  });

  return response.data;
}