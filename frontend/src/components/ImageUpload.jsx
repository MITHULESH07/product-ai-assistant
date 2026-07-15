import "./ImageUpload.css";

function ImageUpload({ image, setImage, imagePreview, setImagePreview }) {
  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;

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
            <span>Attach an image (PNG, JPG, JPEG)</span>
          </div>
        )}
      </label>
      <input
        id="image-input"
        type="file"
        accept=".png,.jpg,.jpeg"
        className="image-upload-input"
        onChange={handleFileChange}
      />
      <p className="image-upload-note">
        Image analysis is not connected in this version.
      </p>
    </div>
  );
}

export default ImageUpload;
