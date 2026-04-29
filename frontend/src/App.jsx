import { useState } from "react";

function App(){


  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);


  function handleFileChange(event) {
    const file = event.target.files[0];
    setSelectedFile(file);
    setResult(null);

    if (file){
      setPreviewUrl(URL.createObjectURL(file));
    }
  }

  async function handlePredict(){

    if (!selectedFile) {
      alert("Please choose an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    setLoading(true)

    try {
      const response = await fetch("http://localhost:8000/predict",{
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed");
      }

      setResult(data);
    } catch (error){
      console.error(error);
      alert(error.message || "Something went wrong. Is the backend running?")
    }

    setLoading(false);
  }
  
  
  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>Potato Disease Detector</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {previewUrl && (
        <div>
          <h3>Selected Image</h3>
          <img
            src={previewUrl}
            alt="Selected potato leaf"
            style={{ width: "300px", marginTop: "10px" }}
          />
        </div>
      )}

      <br />

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Disease"}
      </button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2>Result</h2>
          <p>
            <strong>Disease:</strong> {result.class}
          </p>
          <p>
            <strong>Confidence:</strong>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>

          <h3>All Predictions</h3>
          <ul>
            {Object.entries(result.predictions).map(([name, value]) => (
              <li key={name}>
                {name}: {(value * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
