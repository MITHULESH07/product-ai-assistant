import "./AssistantForm.css";
import ImageUpload from "./ImageUpload";

const PRODUCTS = [
  "ESP32",
  "ESP32-CAM",
  "MG90S Servo",
  "MG996R Servo",
  "PCA9685 Servo Driver",
  "L298N Motor Driver",
  "Robotic Arm",
  "Warehouse Rover",
];

const ASSISTANCE_TYPES = [
  { value: "auto", label: "Auto (Detect)" },
  { value: "operation", label: "Operation" },
  { value: "troubleshooting", label: "Troubleshooting" },
  { value: "maintenance", label: "Maintenance" },
];

function AssistantForm({
  product,
  setProduct,
  assistanceType,
  setAssistanceType,
  question,
  setQuestion,
  image,
  setImage,
  imagePreview,
  setImagePreview,
  onSubmit,
  loading,
  validationMessage,
}) {
  function handleSubmit(e) {
    e.preventDefault();

    if (!question.trim()) {
      return;
    }

    onSubmit();
  }

  return (
    <form className="assistant-form" onSubmit={handleSubmit} noValidate>
      <h2 className="form-heading">Request Assistance</h2>

      <div className="form-field">
        <label className="form-label" htmlFor="product-select">
          Product
        </label>
        <select
          id="product-select"
          className="form-select"
          value={product}
          onChange={(e) => setProduct(e.target.value)}
          disabled={loading}
        >
          {PRODUCTS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>

      <div className="form-field">
        <label className="form-label" htmlFor="assistance-select">
          Assistance Type
        </label>
        <select
          id="assistance-select"
          className="form-select"
          value={assistanceType}
          onChange={(e) => setAssistanceType(e.target.value)}
          disabled={loading}
        >
          {ASSISTANCE_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </div>

      <div className="form-field">
        <label className="form-label" htmlFor="question-input">
          Question <span className="form-required">*</span>
        </label>
        <textarea
          id="question-input"
          className="form-textarea"
          rows={5}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Why is my servo overheating?"
          disabled={loading}
        />
      </div>

      <ImageUpload
        image={image}
        setImage={setImage}
        imagePreview={imagePreview}
        setImagePreview={setImagePreview}
      />

      {validationMessage && (
        <p className="form-validation" role="alert">
          {validationMessage}
        </p>
      )}

      <button
        type="submit"
        className="form-submit"
        disabled={loading}
      >
        {loading ? "Analysing…" : "Analyse Problem"}
      </button>
    </form>
  );
}

export default AssistantForm;
