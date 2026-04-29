import { useState } from "react";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import PredictionResult from "./components/PredictionResult";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleFileChange(event) {
    const file = event.target.files[0];
    setSelectedFile(file);
    setResult(null);

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl("");
    }
  }

  async function handlePredict() {
    if (!selectedFile) {
      alert("Please choose an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed");
      }

      setResult(data);
    } catch (error) {
      console.error(error);
      alert(error.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <div className="app-header">
        <h1>Potato Disease Detector</h1>
        <p>Upload a potato leaf image to classify early blight, late blight, or healthy leaves.</p>
      </div>

      <ImageUploader
        loading={loading}
        onFileChange={handleFileChange}
        onPredict={handlePredict}
        previewUrl={previewUrl}
      />

      <PredictionResult result={result} />
    </main>
  );
}

export default App;
