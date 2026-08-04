import "./LoadingState.css";

function LoadingState() {
  return (
    <div className="loading-state" role="status">
      <div className="loading-spinner" />
      <p className="loading-text">Analysing your problem…</p>
    </div>
  );
}

export default LoadingState;
