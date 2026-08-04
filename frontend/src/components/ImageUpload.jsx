import { useEffect } from "react";
import "./ImageUpload.css";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ACCEPTED_EXTENSIONS = ".png,.jpg,.jpeg,.webp";
const MAX_SIZE_BYTES = 5 * 1024 * 1024;

function ImageUpload({ image, setImage, imagePreview, setImagePreview }) {
  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview);
      }
    };
  }, [imagePreview]);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setImage(null);
      setImagePreview(null);
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setImage(null);
      setImagePreview(null);
      return;
    }

    setImage(file);
    setImagePreview(URL.createObjectURL(file));
  }

  function handleRemove() {
    setImage(null);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImagePreview(null);
  }

  return (
    <div className="image-upload">
      <label className="image-upload-label" htmlFor="image-input">
        {image ? (
          <div className="image-upload-selected">
            <div className="image-upload-preview-wrapper">
              {imagePreview && (
                <img
                  className="image-upload-preview"
                  src={imagePreview}
                  alt="Preview"
                />
              )}
            </div>
            <div className="image-upload-info">
              <span className="image-upload-filename">{image.name}</span>
              <button
                type="button"
                className="image-upload-remove"
                onClick={handleRemove}
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="image-upload-placeholder">
            <span className="image-upload-icon">+</span>
            <span>Attach an image (PNG, JPG, JPEG, WebP)</span>
          </div>
        )}
      </label>
      <input
        id="image-input"
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        className="image-upload-input"
        onChange={handleFileChange}
      />
    </div>
  );
}

export default ImageUpload;
