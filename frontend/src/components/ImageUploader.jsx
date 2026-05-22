function ImageUploader({
  fileInputKey,
  loading,
  onFileChange,
  onPredict,
  onClear,
  previewUrl,
}) {
  return (
    <section className="panel">
      <label className="file-label" htmlFor="leaf-image">
        Choose potato leaf image
      </label>

      <input
        key={fileInputKey}
        id="leaf-image"
        className="file-input"
        type="file"
        accept="image/*"
        onChange={onFileChange}
      />

      {previewUrl && (
        <div className="preview">
          <h2>Selected Image</h2>
          <img src={previewUrl} alt="Selected potato leaf" />
        </div>
      )}

      <button className="predict-button" onClick={onPredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Disease"}
      </button>
      <button className="clear-button" onClick={onClear} disabled={loading}>
        Clear
      </button>
    </section>
  );
}

export default ImageUploader;
