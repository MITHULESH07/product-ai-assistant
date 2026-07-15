import { useState } from "react";
import "./App.css";
import { analyseProblem } from "./services/api.js";
import Header from "./components/Header.jsx";
import AssistantForm from "./components/AssistantForm.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import LoadingState from "./components/LoadingState.jsx";
import ErrorMessage from "./components/ErrorMessage.jsx";

function App() {
  const [product, setProduct] = useState("ESP32");
  const [assistanceType, setAssistanceType] = useState("auto");
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validationMessage, setValidationMessage] = useState("");

  async function handleSubmit() {
    const trimmed = question.trim();
    if (!trimmed) {
      setValidationMessage("Please enter a question.");
      return;
    }

    setValidationMessage("");
    setError("");
    setResult(null);

    try {
      setLoading(true);
      const data = await analyseProblem(trimmed, product, assistanceType);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail ||
          "Could not connect to the backend. Please ensure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function renderResultArea() {
    if (loading) {
      return <LoadingState />;
    }

    if (error) {
      return (
        <div className="result-panel-wrapper">
          <ErrorMessage message={error} />
        </div>
      );
    }

    if (result) {
      return <ResultPanel result={result} />;
    }

    return (
      <div className="result-empty">
        <div className="result-empty-icon">&#9881;</div>
        <p className="result-empty-text">
          Submit a question to receive AI-powered assistance.
        </p>
      </div>
    );
  }

  return (
    <div className="app">
      <Header />

      <div className="dashboard">
        <div className="form-panel">
          <AssistantForm
            product={product}
            setProduct={setProduct}
            assistanceType={assistanceType}
            setAssistanceType={setAssistanceType}
            question={question}
            setQuestion={setQuestion}
            image={image}
            setImage={setImage}
            imagePreview={imagePreview}
            setImagePreview={setImagePreview}
            onSubmit={handleSubmit}
            loading={loading}
            validationMessage={validationMessage}
          />
        </div>

        <div className="result-panel-wrapper">
          {renderResultArea()}
        </div>
      </div>
    </div>
  );
}

export default App;
