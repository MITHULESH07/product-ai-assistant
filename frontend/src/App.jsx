import { useState } from "react";
import "./App.css";
import { analyseProblem } from "./services/api.js";

function App() {
  const [product, setProduct] = useState("ESP32");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const data = await analyseProblem(question, product);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Could not connect to the FastAPI backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: "800px", margin: "40px auto", padding: "20px" }}>
      <h1>AI Product Assistance System</h1>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "16px" }}>
          <label htmlFor="product">Select product</label>
          <br />

          <select
            id="product"
            value={product}
            onChange={(event) => setProduct(event.target.value)}
          >
            <option value="ESP32">ESP32</option>
            <option value="ESP32-CAM">ESP32-CAM</option>
            <option value="MG996R">MG996R Servo</option>
            <option value="MG90S">MG90S Servo</option>
            <option value="PCA9685">PCA9685</option>
            <option value="L298N">L298N Motor Driver</option>
            <option value="Robotic Arm">Robotic Arm</option>
          </select>
        </div>

        <div style={{ marginBottom: "16px" }}>
          <label htmlFor="question">Describe your problem</label>
          <br />

          <textarea
            id="question"
            rows="5"
            style={{ width: "100%" }}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Example: Why is my servo overheating?"
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Analysing..." : "Analyse Problem"}
        </button>
      </form>

      {error && <p>{error}</p>}

      {result && (
        <section style={{ marginTop: "24px" }}>
          <h2>Analysis</h2>

          <p>
            <strong>Intent:</strong> {result.intent}
          </p>

          <p>
            <strong>Product:</strong> {result.product}
          </p>

          <p>
            <strong>Summary:</strong> {result.summary}
          </p>

          <h3>Possible causes</h3>
          <ul>
            {result.possible_causes?.map((cause, index) => (
              <li key={index}>{cause}</li>
            ))}
          </ul>

          <h3>Recommended steps</h3>
          <ol>
            {result.steps?.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>

          {result.warning && (
            <p>
              <strong>Warning:</strong> {result.warning}
            </p>
          )}
        </section>
      )}
    </main>
  );
}

export default App;