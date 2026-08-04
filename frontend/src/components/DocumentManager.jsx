import { useState, useEffect } from "react";
import "./DocumentManager.css";
import {
  ingestDocument,
  getIndexedDocuments,
  deleteIndexedDocument,
} from "../services/api.js";
import { PRODUCTS } from "../constants.js";

const MAX_UPLOAD_SIZE_MB = 20;

function DocumentManager() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [product, setProduct] = useState(PRODUCTS[0]);
  const [replaceExisting, setReplaceExisting] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");
  const [ingestionResult, setIngestionResult] = useState(null);

  const [documents, setDocuments] = useState([]);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [deletingSource, setDeletingSource] = useState(null);
  const [confirmDeleteSource, setConfirmDeleteSource] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoadingDocuments(true);
      setDocumentsError("");
      try {
        const data = await getIndexedDocuments();
        if (!cancelled) {
          setDocuments(data.sources || []);
        }
      } catch (err) {
        if (!cancelled) {
          setDocumentsError(
            err.response?.data?.detail || err.message || "Failed to load documents."
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingDocuments(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  async function refreshDocuments() {
    setLoadingDocuments(true);
    setDocumentsError("");
    try {
      const data = await getIndexedDocuments();
      setDocuments(data.sources || []);
    } catch (err) {
      setDocumentsError(
        err.response?.data?.detail || err.message || "Failed to load documents."
      );
    } finally {
      setLoadingDocuments(false);
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF manuals are supported.");
      setSelectedFile(null);
      e.target.value = "";
      return;
    }

    if (file.size === 0) {
      setError("The selected file is empty.");
      setSelectedFile(null);
      e.target.value = "";
      return;
    }

    if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
      setError(
        `The selected file exceeds the allowed size of ${MAX_UPLOAD_SIZE_MB} MB.`
      );
      setSelectedFile(null);
      e.target.value = "";
      return;
    }

    setError("");
    setSelectedFile(file);
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  async function handleUpload(e) {
    e.preventDefault();

    if (!selectedFile) {
      setError("Please select a PDF file.");
      return;
    }

    setError("");
    setSuccess("");
    setIngestionResult(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await ingestDocument(
        selectedFile,
        replaceExisting
      );
      setUploadProgress(100);
      setIngestionResult(result);
      setSuccess(
        result.message || `Successfully indexed '${result.source}'.`
      );
      setSelectedFile(null);
      refreshDocuments();
    } catch (err) {
      if (err.response) {
        const status = err.response.status;
        const detail =
          err.response.data?.detail || err.response.statusText;

        if (status === 413) {
          setError(
            err.response.data?.detail ||
              "The file exceeds the maximum allowed size."
          );
        } else if (status === 415) {
          setError("Only PDF files are accepted.");
        } else if (status === 422) {
          setError(detail);
        } else if (status === 503) {
          setError(
            "Embedding service unavailable. Please ensure the Ollama server is running."
          );
        } else {
          setError(detail || "Upload failed. Please try again.");
        }
      } else if (err.code === "ERR_NETWORK") {
        setError(
          "Could not connect to the backend. Please ensure the FastAPI server is running."
        );
      } else if (err.code === "ECONNABORTED") {
        setError("The request timed out. Please try again.");
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(source) {
    setConfirmDeleteSource(null);
    setDeletingSource(source);
    setError("");
    setSuccess("");

    try {
      const result = await deleteIndexedDocument(source);
      setSuccess(
        `Deleted ${result.deleted_chunks} chunk(s) from '${result.source}'.`
      );
      refreshDocuments();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Failed to delete document."
      );
    } finally {
      setDeletingSource(null);
    }
  }

  return (
    <div className="document-manager">
      <form className="doc-form" onSubmit={handleUpload} noValidate>
        <h2 className="doc-form-heading">Upload Product Manual</h2>

        <div className="doc-form-field">
          <label className="doc-form-label" htmlFor="doc-product-select">
            Product
          </label>
          <select
            id="doc-product-select"
            className="doc-form-select"
            value={product}
            onChange={(e) => setProduct(e.target.value)}
            disabled={uploading}
          >
            {PRODUCTS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <p className="doc-form-hint">
            Select the product this manual belongs to.
          </p>
        </div>

        <div className="doc-form-field">
          <label className="doc-form-label" htmlFor="doc-file-input">
            PDF Manual
          </label>
          <div className="doc-file-area">
            <input
              id="doc-file-input"
              type="file"
              accept="application/pdf,.pdf"
              className="doc-file-input"
              onChange={handleFileChange}
              disabled={uploading}
            />
            {selectedFile ? (
              <div className="doc-file-info">
                <span className="doc-file-name">
                  {selectedFile.name}
                </span>
                <span className="doc-file-size">
                  {formatFileSize(selectedFile.size)}
                </span>
              </div>
            ) : (
              <p className="doc-file-placeholder">
                Click to select a PDF file
              </p>
            )}
          </div>
        </div>

        <div className="doc-form-field doc-form-field--checkbox">
          <label className="doc-form-checkbox-label">
            <input
              type="checkbox"
              className="doc-form-checkbox"
              checked={replaceExisting}
              onChange={(e) => setReplaceExisting(e.target.checked)}
              disabled={uploading}
            />
            <span>Replace existing chunks for this manual</span>
          </label>
        </div>

        {uploading && (
          <div className="doc-upload-progress" role="status">
            <div className="doc-progress-bar-track">
              <div
                className="doc-progress-bar-fill"
                style={{ width: uploadProgress + "%" }}
              />
            </div>
            <p className="doc-progress-text">
              {uploadProgress < 100
                ? "Uploading…"
                : "Upload complete. Indexing document…"}
            </p>
          </div>
        )}

        {error && (
          <p className="doc-error" role="alert">
            {error}
          </p>
        )}

        {success && (
          <p className="doc-success" role="status">
            {success}
          </p>
        )}

        {ingestionResult && (
          <div className="doc-ingestion-result">
            <h3 className="doc-ingestion-heading">Indexing Result</h3>
            <dl className="doc-ingestion-details">
              <dt>Source</dt>
              <dd>{ingestionResult.source}</dd>
              <dt>Pages extracted</dt>
              <dd>{ingestionResult.pages_extracted}</dd>
              <dt>Chunks created</dt>
              <dd>{ingestionResult.chunks_created}</dd>
              <dt>Chunks stored</dt>
              <dd>{ingestionResult.chunks_stored}</dd>
              {ingestionResult.previous_chunks_deleted > 0 && (
                <>
                  <dt>Previous chunks replaced</dt>
                  <dd>{ingestionResult.previous_chunks_deleted}</dd>
                </>
              )}
              <dt>Collection count</dt>
              <dd>{ingestionResult.collection_count}</dd>
            </dl>
          </div>
        )}

        <button
          type="submit"
          className="doc-form-submit"
          disabled={uploading || !selectedFile}
        >
          {uploading ? "Uploading…" : "Upload & Index"}
        </button>
      </form>

      <section className="doc-list-section">
        <div className="doc-list-header">
          <h2 className="doc-list-heading">Indexed Manuals</h2>
          <button
            type="button"
            className="doc-refresh-btn"
            onClick={refreshDocuments}
            disabled={loadingDocuments}
            aria-label="Refresh document list"
          >
            Refresh
          </button>
        </div>

        {loadingDocuments && (
          <p className="doc-list-status" role="status">
            Loading documents…
          </p>
        )}

        {documentsError && (
          <p className="doc-error" role="alert">
            {documentsError}
          </p>
        )}

        {!loadingDocuments && !documentsError && documents.length === 0 && (
          <p className="doc-list-empty">
            No manuals indexed yet. Upload a PDF above.
          </p>
        )}

        {!loadingDocuments && documents.length > 0 && (
          <ul className="doc-list">
            {documents.map((source) => (
              <li key={source} className="doc-list-item">
                <span className="doc-list-source">{source}</span>

                {confirmDeleteSource === source ? (
                  <span className="doc-delete-confirm">
                    <span className="doc-delete-confirm-text">
                      Delete this indexed manual?
                    </span>
                    <button
                      type="button"
                      className="doc-delete-yes"
                      onClick={() => handleDelete(source)}
                      disabled={deletingSource === source}
                    >
                      {deletingSource === source
                        ? "Deleting…"
                        : "Yes, delete"}
                    </button>
                    <button
                      type="button"
                      className="doc-delete-no"
                      onClick={() => setConfirmDeleteSource(null)}
                      disabled={deletingSource === source}
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    className="doc-delete-btn"
                    onClick={() => setConfirmDeleteSource(source)}
                    disabled={deletingSource === source}
                    aria-label={`Delete ${source}`}
                  >
                    {deletingSource === source ? "…" : "Delete"}
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export default DocumentManager;
