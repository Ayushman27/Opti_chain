import React from "react";

export default function InventoryChart({ stepHistory = [], currentStep = 0 }) {
  if (!stepHistory || stepHistory.length === 0) {
    return <div className="glass-panel chart-placeholder">Loading simulation data...</div>;
  }

  const width = 800;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 30, left: 50 };

  const maxOnHand = Math.max(...stepHistory.map((d) => d.on_hand_units || 0), 4000);
  const maxDemand = Math.max(...stepHistory.map((d) => d.demand || 0), 800);

  const getX = (step) =>
    padding.left + (step / Math.max(1, stepHistory.length - 1)) * (width - padding.left - padding.right);
  const getYStock = (val) =>
    height - padding.bottom - (val / maxOnHand) * (height - padding.top - padding.bottom);
  const getYDemand = (val) =>
    height - padding.bottom - (val / maxDemand) * (height - padding.top - padding.bottom);

  // Generate SVG path for on-hand stock
  const stockPoints = stepHistory.map((d, i) => `${getX(i)},${getYStock(d.on_hand_units)}`).join(" ");
  // Generate SVG path for demand
  const demandPoints = stepHistory.map((d, i) => `${getX(i)},${getYDemand(d.demand)}`).join(" ");

  // Current day pointer
  const curX = getX(currentStep);

  return (
    <div className="glass-panel chart-card">
      <div className="chart-header">
        <div>
          <h3 className="section-title">Multi-Period Inventory & Demand Dynamics</h3>
          <p className="section-desc">
            On-hand inventory level vs realized customer demand and stockout vulnerability zones
          </p>
        </div>
        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-color cyan"></span>
            <span>On-Hand Inventory (Units)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color amber"></span>
            <span>Customer Demand</span>
          </div>
          <div className="legend-item">
            <span className="legend-color rose"></span>
            <span>Disruption Windows</span>
          </div>
        </div>
      </div>

      <div className="chart-svg-wrapper">
        <svg viewBox={`0 0 ${width} ${height}`} className="main-chart-svg">
          <defs>
            <linearGradient id="stockGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00f2fe" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#00f2fe" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="disruptionGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.03" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0.25, 0.5, 0.75].map((pct, idx) => {
            const y = height - padding.bottom - pct * (height - padding.top - padding.bottom);
            return (
              <line
                key={idx}
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="rgba(255,255,255,0.06)"
                strokeDasharray="4 4"
              />
            );
          })}

          {/* Disruption Shading Bands: Day 25-32 & Day 45-57 & Day 68-76 */}
          <rect
            x={getX(25)}
            y={padding.top}
            width={getX(32) - getX(25)}
            height={height - padding.top - padding.bottom}
            fill="url(#disruptionGrad)"
          />
          <rect
            x={getX(45)}
            y={padding.top}
            width={getX(57) - getX(45)}
            height={height - padding.top - padding.bottom}
            fill="url(#disruptionGrad)"
          />
          <rect
            x={getX(68)}
            y={padding.top}
            width={getX(76) - getX(68)}
            height={height - padding.top - padding.bottom}
            fill="url(#disruptionGrad)"
          />

          {/* Stock Area fill */}
          <polygon
            points={`${padding.left},${height - padding.bottom} ${stockPoints} ${getX(
              stepHistory.length - 1
            )},${height - padding.bottom}`}
            fill="url(#stockGradient)"
          />

          {/* Demand Polyline */}
          <polyline
            points={demandPoints}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="1.8"
            strokeDasharray="3 3"
          />

          {/* Stock Polyline */}
          <polyline
            points={stockPoints}
            fill="none"
            stroke="#00f2fe"
            strokeWidth="2.5"
            strokeLinecap="round"
          />

          {/* Scrubber indicator line */}
          <line
            x1={curX}
            y1={padding.top}
            x2={curX}
            y2={height - padding.bottom}
            stroke="#f1f5f9"
            strokeWidth="1.5"
            strokeDasharray="2 2"
          />
          <circle
            cx={curX}
            cy={getYStock(stepHistory[currentStep]?.on_hand_units || 0)}
            r="4.5"
            fill="#00f2fe"
            stroke="#fff"
            strokeWidth="2"
          />
        </svg>
      </div>

      <div className="chart-stats-ribbon">
        <div className="ribbon-item">
          <span className="ribbon-label">Day {currentStep} Stock:</span>
          <span className="ribbon-val cyan">
            {(stepHistory[currentStep]?.on_hand_units || 0).toLocaleString()} units
          </span>
        </div>
        <div className="ribbon-item">
          <span className="ribbon-label">Customer Demand:</span>
          <span className="ribbon-val amber">
            {(stepHistory[currentStep]?.demand || 0).toLocaleString()} units
          </span>
        </div>
        <div className="ribbon-item">
          <span className="ribbon-label">Step Fill Rate:</span>
          <span className="ribbon-val emerald">
            {stepHistory[currentStep]?.fill_rate || 100}%
          </span>
        </div>
      </div>
    </div>
  );
}
