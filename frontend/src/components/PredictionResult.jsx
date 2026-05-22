function PredictionResult({ result }) {
  if (!result) {
    return null;
  }

  return (
    <section className="panel result-panel">
      <h2>Result</h2>

      <div className="main-result">
        <span>Disease</span>
        <strong>{result.class}</strong>
      </div>

      <div className="main-result">
        <span>Confidence</span>
        <strong>{(result.confidence * 100).toFixed(2)}%</strong>
      </div>

      <h3>All Predictions</h3>
      <ul className="prediction-list">
        {Object.entries(result.predictions).map(([name, value]) => (
          <li key={name}>
            <span>{name}</span>
            <strong>{(value * 100).toFixed(2)}%</strong>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default PredictionResult;
