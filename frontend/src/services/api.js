import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 180000,
});

export async function analyseProblem(question, product, assistanceType = "auto") {
  const response = await api.post("/analyse", {
    question,
    product,
    assistance_type: assistanceType,
  });

  return response.data;
}