import React, { useState, useEffect } from "react";
import {
  BarChart3,
  TrendingDown,
  Target,
  Zap,
  Layers,
  RefreshCw,
  AlertCircle,
  ShieldCheck,
  Calendar,
} from "lucide-react";
import { fetchForecasts } from "../services/api";

export default function ForecastView() {
  const [productId, setProductId] = useState("PROD_001");
  const [model, setModel] = useState("Ridge");
  const [horizon, setHorizon] = useState(14);

  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchForecasts({
        productId,
        model,
        horizon,
      });
      setForecastData(data);
    } catch (err) {
      console.error("Forecast fetch error:", err);
      setError(err.message || "Failed to load forecast data from backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadForecast();
  }, [productId, model, horizon]);

  const availableModels = forecastData?.available_models || [
    "Ridge",
    "LightGBM",
    "Seasonal_Naive_7d",
    "Moving_Average_7d",
  ];
  const availableProducts = forecastData?.available_products || [
    "PROD_001",
    "PROD_002",
    "PROD_003",
    "PROD_004",
    "PROD_005",
  ];

  const metrics = forecastData?.metrics || {
    wape_percent: 0,
    mae: 0,
    rmse: 0,
    bias_percent: 0,
  };

  // Render SVG time-series chart
  const renderChart = () => {
    if (!forecastData) return null;

    const histSteps = forecastData.historical_steps || [];
    const histActual = forecastData.historical_actual || [];
    const evalSteps = forecastData.evaluation_steps || [];
    const evalActual = forecastData.evaluation_actual || [];
    const evalPreds = forecastData.evaluation_predictions || [];
    const futureSteps = forecastData.future_steps || [];
    const futurePreds = forecastData.future_predictions || [];

    const allValues = [
      ...histActual,
      ...evalActual,
      ...evalPreds,
      ...futurePreds,
    ];
    const maxVal = Math.max(...allValues, 40);
    const minVal = 0;

    const width = 800;
    const height = 260;
    const padding = { top: 30, right: 30, bottom: 40, left: 50 };

    const totalPoints = histSteps.length + evalSteps.length + futureSteps.length;
    const getX = (idx) =>
      padding.left + (idx / Math.max(1, totalPoints - 1)) * (width - padding.left - padding.right);
    const getY = (val) =>
      height - padding.bottom - ((val - minVal) / Math.max(1, maxVal - minVal)) * (height - padding.top - padding.bottom);

    // Build historical points
    const histPts = histActual.map((v, i) => `${getX(i)},${getY(v)}`).join(" ");

    // Build evaluation actual points (offset by histSteps.length)
    const evalActualPts = evalActual.map((v, i) => `${getX(histSteps.length + i)},${getY(v)}`).join(" ");

    // Build evaluation prediction points
    const evalPredPts = evalPreds.map((v, i) => `${getX(histSteps.length + i)},${getY(v)}`).join(" ");

    // Build future prediction points (offset by histSteps + evalSteps)
    const futurePredPts = futurePreds.map(
      (v, i) => `${getX(histSteps.length + evalSteps.length + i)},${getY(v)}`
    ).join(" ");

    // Boundary X coordinates
    const evalStartX = getX(histSteps.length);
    const futureStartX = getX(histSteps.length + evalSteps.length);

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="main-chart-svg" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id="evalZoneGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.01" />
          </linearGradient>
          <linearGradient id="futureZoneGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {/* Region Backgrounds */}
        {/* Evaluation Window Background */}
        <rect
          x={evalStartX}
          y={padding.top}
          width={futureStartX - evalStartX}
          height={height - padding.top - padding.bottom}
          fill="url(#evalZoneGrad)"
        />
        {/* Future Horizon Window Background */}
        <rect
          x={futureStartX}
          y={padding.top}
          width={width - padding.right - futureStartX}
          height={height - padding.top - padding.bottom}
          fill="url(#futureZoneGrad)"
        />

        {/* Gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
          const y = height - padding.bottom - pct * (height - padding.top - padding.bottom);
          const val = Math.round(minVal + pct * (maxVal - minVal));
          return (
            <g key={i}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <text x={padding.left - 8} y={y + 4} textAnchor="end" fill="var(--text-dim)" fontSize="10" fontFamily="JetBrains Mono">
                {val}
              </text>
            </g>
          );
        })}

        {/* Vertical Region Partition Dividers */}
        <line x1={evalStartX} y1={padding.top} x2={evalStartX} y2={height - padding.bottom} stroke="rgba(59, 130, 246, 0.5)" strokeDasharray="4 2" />
        <text x={evalStartX + 4} y={padding.top + 12} fill="#3b82f6" fontSize="10" fontWeight="700">
          Evaluation Split (Day 63)
        </text>

        <line x1={futureStartX} y1={padding.top} x2={futureStartX} y2={height - padding.bottom} stroke="rgba(16, 185, 129, 0.5)" strokeDasharray="4 2" />
        <text x={futureStartX + 4} y={padding.top + 12} fill="#10b981" fontSize="10" fontWeight="700">
          Forward Forecast
        </text>

        {/* 1. Historical Actual Line (Past) */}
        {histPts && (
          <polyline points={histPts} fill="none" stroke="var(--text-muted)" strokeWidth="1.8" />
        )}

        {/* 2. Out-of-sample Evaluation Actuals Line */}
        {evalActualPts && (
          <polyline points={evalActualPts} fill="none" stroke="#60a5fa" strokeWidth="2.2" />
        )}

        {/* 3. Out-of-sample Model Predictions Line */}
        {evalPredPts && (
          <polyline points={evalPredPts} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 3" />
        )}

        {/* 4. Future Out-of-sample Forecast Line */}
        {futurePredPts && (
          <polyline points={futurePredPts} fill="none" stroke="#10b981" strokeWidth="2.5" />
        )}

        {/* X-axis labels */}
        <text x={padding.left} y={height - 12} fill="var(--text-dim)" fontSize="10">
          Day {histSteps[0] || 0} (Training)
        </text>
        <text x={evalStartX} y={height - 12} fill="#60a5fa" fontSize="10">
          Day {evalSteps[0] || 63} (Test)
        </text>
        <text x={futureStartX} y={height - 12} fill="#10b981" fontSize="10">
          Day {futureSteps[0] || 77} (Future)
        </text>
      </svg>
    );
  };

  return (
    <div className="forecast-page">
      {/* Header Banner */}
      <div className="forecast-header glass-panel">
        <div className="banner-badge">
          <BarChart3 size={18} />
          <span>Demand Forecasting & Rolling Validation Pipeline</span>
        </div>
        <h2 className="banner-title">Machine Learning Demand Projections</h2>
        <p className="banner-subtitle">
          Evaluated via strict chronological out-of-time splits. Zero leakage into future evaluation steps.
        </p>

        {/* Parameter Controls Bar */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", marginTop: "18px", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Target SKU:</span>
            <select
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              className="glass-input"
              style={{ padding: "6px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            >
              {availableProducts.map((p) => (
                <option key={p} value={p} style={{ background: "#0c1019", color: "#fff" }}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Forecasting Model:</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="glass-input"
              style={{ padding: "6px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            >
              {availableModels.map((m) => (
                <option key={m} value={m} style={{ background: "#0c1019", color: "#fff" }}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Horizon:</span>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="glass-input"
              style={{ padding: "6px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            >
              <option value={7} style={{ background: "#0c1019", color: "#fff" }}>7 Days</option>
              <option value={14} style={{ background: "#0c1019", color: "#fff" }}>14 Days</option>
              <option value={21} style={{ background: "#0c1019", color: "#fff" }}>21 Days</option>
              <option value={27} style={{ background: "#0c1019", color: "#fff" }}>27 Days</option>
            </select>
          </div>

          <button
            onClick={loadForecast}
            disabled={loading}
            className="btn-step"
            style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "6px" }}
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} /> Refresh Forecast
          </button>
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="glass-panel state-card" style={{ marginBottom: "20px" }}>
          <RefreshCw className="animate-spin icon-spin" size={28} />
          <h3 className="state-title">Generating Forecast for {productId}...</h3>
        </div>
      )}

      {error && (
        <div className="glass-panel state-card error-card" style={{ marginBottom: "20px" }}>
          <AlertCircle size={32} className="icon-error" />
          <h3 className="state-title">Forecast Generation Failed</h3>
          <p className="state-desc">{error}</p>
          <button onClick={loadForecast} className="btn-playback btn-play" style={{ marginTop: "12px" }}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && forecastData && (
        <>
          {/* Evaluation Scope & Benchmark Provenance Banner */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", gap: "10px" }}>
            <span className="badge-pill cyan" style={{ fontSize: "12px" }}>
              Active Evaluation: {forecastData.evaluation_scope || `${forecastData.product_id} (${forecastData.horizon_days} days)`}
            </span>
            {forecastData.benchmark_metrics && (
              <span className="badge-pill purple" style={{ fontSize: "11px" }}>
                Benchmark Reference: Days 63-89 across all 5 SKUs (135 obs)
              </span>
            )}
          </div>

          {/* Accuracy Scorecards */}
          <div className="forecast-metrics-grid">
            <div className="glass-panel metric-card">
              <div className="metric-header">
                <span>Weighted Absolute Error (WAPE)</span>
                <Target size={16} className="cyan" />
              </div>
              <div className="metric-val cyan">{metrics.wape_percent}%</div>
              <div className="metric-compare">
                {forecastData.benchmark_metrics ? (
                  <span>Overall 27-day Benchmark: <strong style={{ color: "var(--accent-cyan)" }}>{forecastData.benchmark_metrics.wape_percent}%</strong> (All SKUs)</span>
                ) : (
                  "Total absolute error / total demand"
                )}
              </div>
            </div>

            <div className="glass-panel metric-card">
              <div className="metric-header">
                <span>Mean Absolute Error (MAE)</span>
                <TrendingDown size={16} className="emerald" />
              </div>
              <div className="metric-val emerald">{metrics.mae} units</div>
              <div className="metric-compare">
                {forecastData.benchmark_metrics ? (
                  <span>Overall 27-day Benchmark: <strong style={{ color: "var(--accent-emerald)" }}>{forecastData.benchmark_metrics.mae} units</strong></span>
                ) : (
                  "Mean unit deviation per day"
                )}
              </div>
            </div>

            <div className="glass-panel metric-card">
              <div className="metric-header">
                <span>Root Mean Squared Error (RMSE)</span>
                <Zap size={16} className="amber" />
              </div>
              <div className="metric-val amber">{metrics.rmse}</div>
              <div className="metric-compare">
                {forecastData.benchmark_metrics ? (
                  <span>Overall 27-day Benchmark: <strong style={{ color: "var(--accent-amber)" }}>{forecastData.benchmark_metrics.rmse}</strong></span>
                ) : (
                  "Penalizes large forecast errors"
                )}
              </div>
            </div>

            <div className="glass-panel metric-card">
              <div className="metric-header">
                <span>Forecast Bias</span>
                <Layers size={16} className="rose" />
              </div>
              <div className="metric-val rose">{metrics.bias_percent}%</div>
              <div className="metric-compare">
                {forecastData.benchmark_metrics ? (
                  <span>Overall 27-day Benchmark: <strong style={{ color: "var(--accent-rose)" }}>{forecastData.benchmark_metrics.bias_percent}%</strong></span>
                ) : (
                  metrics.bias_percent < 0 ? "Under-forecasting tendency" : "Over-forecasting tendency"
                )}
              </div>
            </div>
          </div>

          {/* Time Series Chart */}
          <div className="glass-panel chart-card" style={{ marginTop: "24px" }}>
            <div className="chart-header">
              <div>
                <h3 className="section-title">
                  {forecastData.model_name} Multi-Horizon Projections: {forecastData.product_id}
                </h3>
                <p className="section-desc">
                  Chronological progression: Historical Training Actuals $\to$ Out-of-Sample Evaluation $\to$ Forward Forecast
                </p>
              </div>

              <div className="chart-legend" style={{ flexWrap: "wrap" }}>
                <div className="legend-item">
                  <span className="legend-color" style={{ background: "var(--text-muted)" }}></span>
                  <span>Historical Actuals</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color" style={{ background: "#60a5fa" }}></span>
                  <span>Evaluation Actuals</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color" style={{ background: "#f59e0b" }}></span>
                  <span>Model Predicted</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color" style={{ background: "#10b981" }}></span>
                  <span>Future Forward Forecast</span>
                </div>
              </div>
            </div>

            <div className="chart-svg-wrapper">
              {renderChart()}
            </div>

            <div className="chart-stats-ribbon" style={{ marginTop: "16px", justifyContent: "space-between" }}>
              <div className="ribbon-item">
                <ShieldCheck size={14} className="cyan" style={{ marginRight: "4px" }} />
                <span className="ribbon-label">Integrity Status:</span>
                <span className="ribbon-val cyan">{forecastData.leakage_audit_status}</span>
              </div>
              <div className="ribbon-item">
                <Calendar size={14} className="amber" style={{ marginRight: "4px" }} />
                <span className="ribbon-label">Horizon Duration:</span>
                <span className="ribbon-val amber">{forecastData.horizon_days} steps</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
