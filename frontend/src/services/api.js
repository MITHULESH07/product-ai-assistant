import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 600000,
});

export async function analyseProblem(
  question,
  product,
  assistanceType = "troubleshooting",
  image = null
) {
  if (image) {
    const formData = new FormData();
    formData.append("question", question);
    formData.append("product", product);
    formData.append("assistance_type", assistanceType);
    formData.append("file", image);

    const response = await api.post("/analyse", formData);
    return response.data;
  }

  const response = await api.post("/analyse", {
    question,
    product,
    assistance_type: assistanceType,
  });

  return response.data;
}

export async function ingestDocument(file, replaceExisting = true) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("replace_existing", String(replaceExisting));

  const response = await api.post("/documents/ingest", formData);
  return response.data;
}

export async function getIndexedDocuments() {
  const response = await api.get("/documents");
  return response.data;
}

export async function deleteIndexedDocument(source) {
  const response = await api.delete(`/documents/${encodeURIComponent(source)}`);
  return response.data;
}
