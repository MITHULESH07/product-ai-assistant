import "./ResultPanel.css";

const ASSISTANCE_LABELS = {
  auto: "Auto",
  operation: "Operation",
  troubleshooting: "Troubleshooting",
  maintenance: "Maintenance",
};

function ResultPanel({ result }) {
  if (!result) return null;

  const {
    intent,
    product,
    summary,
    possible_causes,
    steps,
    warning,
    escalation_required,
    sources,
    assistance_type,
  } = result;

  const causes = Array.isArray(possible_causes) ? possible_causes : [];
  const stepList = Array.isArray(steps) ? steps : [];
  const sourceList = Array.isArray(sources) ? sources : [];

  return (
    <section className="result-panel">
      <h2 className="result-heading">Analysis Result</h2>

      <div className="result-badges">
        {intent && <span className="badge badge-intent">{intent}</span>}
        {product && <span className="badge badge-product">{product}</span>}
        {assistance_type && (
          <span className="badge badge-type">
            {ASSISTANCE_LABELS[assistance_type] || assistance_type}
          </span>
        )}
      </div>

      {summary && (
        <div className="result-section">
          <h3 className="result-section-title">Summary</h3>
          <p className="result-summary">{summary}</p>
        </div>
      )}

      {causes.length > 0 && (
        <div className="result-section">
          <h3 className="result-section-title">Possible Causes</h3>
          <ul className="result-list">
            {causes.map((cause, i) => (
              <li key={i} className="result-list-item">
                {cause}
              </li>
            ))}
          </ul>
        </div>
      )}

      {stepList.length > 0 && (
        <div className="result-section">
          <h3 className="result-section-title">Recommended Steps</h3>
          <ol className="result-list result-list--ordered">
            {stepList.map((step, i) => (
              <li key={i} className="result-list-item">
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      {warning && (
        <div className="result-section result-section--warning">
          <h3 className="result-section-title">Warning</h3>
          <p className="result-warning">{warning}</p>
        </div>
      )}

      {escalation_required !== undefined && (
        <div className="result-section">
          <h3 className="result-section-title">Escalation Status</h3>
          <p className="result-escalation">
            {escalation_required
              ? "Escalation required — expert intervention needed."
              : "No escalation required."}
          </p>
        </div>
      )}

      {sourceList.length > 0 && (
        <div className="result-section">
          <h3 className="result-section-title">Sources</h3>
          <ul className="result-sources">
            {sourceList.map((src, i) => (
              <li key={i} className="result-source-item">
                {src.document && src.page != null
                  ? `${src.document} — Page ${src.page}`
                  : typeof src === "string"
                    ? src
                    : JSON.stringify(src)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export default ResultPanel;
