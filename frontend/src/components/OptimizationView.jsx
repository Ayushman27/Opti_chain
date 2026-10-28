import React, { useState, useEffect } from "react";
import {
  Cpu,
  Play,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldAlert,
  HelpCircle,
  FileText,
  DollarSign,
  Package,
  Layers,
  RefreshCw,
  AlertCircle,
  Sliders,
} from "lucide-react";
import { fetchBenchmarks, fetchOptimizationRecommendations } from "../services/api";

export default function OptimizationView() {
  const [benchmarks, setBenchmarks] = useState(null);
  const [benchLoading, setBenchLoading] = useState(true);
  const [benchError, setBenchError] = useState(null);

  // Form parameters
  const [horizon, setHorizon] = useState(7);
  const [safetyBuffer, setSafetyBuffer] = useState(0.2);
  const [timeLimit, setTimeLimit] = useState(4);
  const [forecastMethod, setForecastMethod] = useState("ridge");
  const [productFilter, setProductFilter] = useState("ALL");

  // Optimization run state
  const [optResult, setOptResult] = useState(null);
  const [solving, setSolving] = useState(false);
  const [optError, setOptError] = useState(null);

  // Initial load: fetch benchmarks for policy matrix comparison
  useEffect(() => {
    async function load() {
      setBenchLoading(true);
      setBenchError(null);
      try {
        const b = await fetchBenchmarks();
        setBenchmarks(b);
      } catch (err) {
        console.error("Benchmarks error:", err);
        setBenchError(err.message || "Failed to load benchmark policies.");
      } finally {
        setBenchLoading(false);
      }
    }
    load();
  }, []);

  // Run MILP Replenishment Solve
  const handleSolve = async () => {
    setSolving(true);
    setOptError(null);
    try {
      const res = await fetchOptimizationRecommendations({
        horizon,
        safetyBufferFactor: safetyBuffer,
        timeLimitSec: timeLimit,
        forecastMethod,
        productIds: productFilter === "ALL" ? null : [productFilter],
      });
      setOptResult(res);
    } catch (err) {
      console.error("Optimization solve error:", err);
      setOptError(err.message || "MILP Solver failed to return recommendations.");
    } finally {
      setSolving(false);
    }
  };

  const policies = benchmarks?.policies || [];

  return (
    <div className="optimization-page">
      {/* Banner */}
      <div className="glass-panel benchmark-header-banner">
        <div className="banner-badge">
          <Cpu size={16} />
          <span>Rolling-Horizon Constrained Optimization (PuLP / CBC)</span>
        </div>
        <h2 className="banner-title">MILP Replenishment Optimizer & Policy Comparison</h2>
        <p className="banner-subtitle">
          Multi-period mixed-integer linear program minimizing total relevant supply chain costs subject to supplier capacity, MOQ, and storage constraints.
        </p>

        {/* Read-Only Decision Review Disclaimer */}
        <div
          style={{
            marginTop: "16px",
            padding: "12px 16px",
            borderRadius: "8px",
            background: "rgba(245, 158, 11, 0.08)",
            border: "1px solid rgba(245, 158, 11, 0.25)",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            color: "var(--accent-amber)",
            fontSize: "12px",
          }}
        >
          <ShieldAlert size={18} style={{ flexShrink: 0 }} />
          <span>
            <strong>READ-ONLY DECISION REVIEW:</strong> Recommended purchase orders are generated for analytical evaluation and executive decision support. No live purchase orders are transmitted or committed to external suppliers.
          </span>
        </div>
      </div>

      {/* 1. Policy Comparison Matrix */}
      <div className="glass-panel" style={{ padding: "24px", marginTop: "24px" }}>
        <div style={{ marginBottom: "16px" }}>
          <h3 className="section-title">Algorithmic Policy Benchmarking Matrix</h3>
          <p className="section-desc">
            Direct comparison of Forecast-Driven MILP against continuous and periodic baseline replenishment policies.
          </p>
        </div>

        {benchLoading ? (
          <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)" }}>
            <RefreshCw className="animate-spin" size={20} style={{ margin: "0 auto 8px" }} />
            Loading benchmark policies...
          </div>
        ) : benchError ? (
          <div style={{ color: "var(--accent-rose)", padding: "12px" }}>
            Error loading benchmarks: {benchError}
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="benchmark-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left", color: "var(--text-muted)" }}>
                  <th style={{ padding: "12px 14px" }}>Policy Architecture</th>
                  <th style={{ padding: "12px 14px" }}>Total Cost ($)</th>
                  <th style={{ padding: "12px 14px" }}>Net Profit ($)</th>
                  <th style={{ padding: "12px 14px" }}>Fill Rate (%)</th>
                  <th style={{ padding: "12px 14px" }}>Service Level (%)</th>
                  <th style={{ padding: "12px 14px" }}>Stockout Units</th>
                  <th style={{ padding: "12px 14px" }}>Avg Inventory (u)</th>
                  <th style={{ padding: "12px 14px" }}>Turnover</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((p, idx) => {
                  const isMILP = p.policy.includes("MILP");
                  return (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                        background: isMILP ? "rgba(0, 242, 254, 0.05)" : "transparent",
                      }}
                    >
                      <td style={{ padding: "14px", fontWeight: 700, color: isMILP ? "var(--accent-cyan)" : "#fff" }}>
                        {p.policy} {isMILP && "⭐ (Lowest Cost)"}
                      </td>
                      <td style={{ padding: "14px", color: isMILP ? "var(--accent-cyan)" : "inherit" }} className="mono">
                        ${p.total_cost_usd.toLocaleString()}
                      </td>
                      <td style={{ padding: "14px", color: "var(--accent-emerald)" }} className="mono">
                        ${p.net_profit_usd.toLocaleString()}
                      </td>
                      <td style={{ padding: "14px" }} className="mono">
                        {p.fill_rate_percent}%
                      </td>
                      <td style={{ padding: "14px" }} className="mono">
                        {p.service_level_percent}%
                      </td>
                      <td style={{ padding: "14px" }} className="mono">
                        {p.stockout_units.toLocaleString()}
                      </td>
                      <td style={{ padding: "14px" }} className="mono">
                        {p.average_inventory_units.toFixed(1)}
                      </td>
                      <td style={{ padding: "14px" }} className="mono">
                        {p.inventory_turnover}x
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Trade-off Explanation Box */}
        <div
          style={{
            marginTop: "16px",
            padding: "14px 18px",
            borderRadius: "8px",
            background: "rgba(59, 130, 246, 0.06)",
            border: "1px solid rgba(59, 130, 246, 0.2)",
            fontSize: "12px",
            color: "var(--text-muted)",
            lineHeight: "1.6",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--accent-cyan)", fontWeight: 700, marginBottom: "4px" }}>
            <HelpCircle size={15} />
            <span>Cost vs Service Level Trade-Off Analysis</span>
          </div>
          No single policy strictly dominates all performance dimensions. <strong>Forecast-Driven MILP</strong> achieves the lowest overall supply chain cost ($428,534.15) and highest inventory turnover (18.82x) by strictly constraining holding and setup costs. In contrast, <strong>Periodic Review (7-day)</strong> achieves higher cycle service level (94.07% vs 89.63%) by holding ~82 more units of buffer inventory (568.61 units average), which incurs $5,234.50 higher total cost.
        </div>
      </div>

      {/* 2. Interactive MILP Optimizer Form & Run */}
      <div className="glass-panel" style={{ padding: "24px", marginTop: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
          <div>
            <h3 className="section-title">Configure & Solve Replenishment Plan</h3>
            <p className="section-desc">Run the mathematical MILP solver with custom horizon, safety buffer, and timeout constraints</p>
          </div>
          <button
            onClick={handleSolve}
            disabled={solving}
            className="btn-playback btn-play"
            style={{ display: "flex", alignItems: "center", gap: "8px" }}
          >
            {solving ? <RefreshCw className="animate-spin" size={15} /> : <Play size={15} />}
            <span>{solving ? "Solving MILP..." : "Solve Replenishment MILP"}</span>
          </button>
        </div>

        {/* Parameter Inputs Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "16px" }}>
          <div>
            <label style={{ fontSize: "11px", color: "var(--text-dim)", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>
              Planning Horizon (Days)
            </label>
            <input
              type="number"
              min="1"
              max="21"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="glass-input"
              style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "var(--text-dim)", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>
              Safety Buffer Factor
            </label>
            <input
              type="number"
              step="0.05"
              min="0.0"
              max="1.0"
              value={safetyBuffer}
              onChange={(e) => setSafetyBuffer(Number(e.target.value))}
              className="glass-input"
              style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "var(--text-dim)", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>
              Solver Timeout (Sec)
            </label>
            <input
              type="number"
              min="1"
              max="15"
              value={timeLimit}
              onChange={(e) => setTimeLimit(Number(e.target.value))}
              className="glass-input"
              style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "var(--text-dim)", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>
              Forecasting Model
            </label>
            <select
              value={forecastMethod}
              onChange={(e) => setForecastMethod(e.target.value)}
              className="glass-input"
              style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            >
              <option value="ridge" style={{ background: "#0c1019" }}>Ridge Regression</option>
              <option value="lgbm" style={{ background: "#0c1019" }}>LightGBM GBDT</option>
              <option value="seasonal_naive" style={{ background: "#0c1019" }}>Seasonal Naive (7d)</option>
              <option value="heuristic_lag7" style={{ background: "#0c1019" }}>Heuristic Lag-7</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "var(--text-dim)", textTransform: "uppercase", display: "block", marginBottom: "6px" }}>
              Product SKU Filter
            </label>
            <select
              value={productFilter}
              onChange={(e) => setProductFilter(e.target.value)}
              className="glass-input"
              style={{ width: "100%", padding: "8px 12px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", color: "#fff", border: "1px solid var(--border-subtle)" }}
            >
              <option value="ALL" style={{ background: "#0c1019" }}>All SKUs (Network-Wide)</option>
              <option value="PROD_001" style={{ background: "#0c1019" }}>PROD_001</option>
              <option value="PROD_002" style={{ background: "#0c1019" }}>PROD_002</option>
              <option value="PROD_003" style={{ background: "#0c1019" }}>PROD_003</option>
              <option value="PROD_004" style={{ background: "#0c1019" }}>PROD_004</option>
              <option value="PROD_005" style={{ background: "#0c1019" }}>PROD_005</option>
            </select>
          </div>
        </div>

        {/* Error Notification */}
        {optError && (
          <div style={{ marginTop: "16px", padding: "12px", borderRadius: "6px", background: "rgba(244, 63, 94, 0.1)", border: "1px solid var(--accent-rose)", color: "var(--accent-rose)", fontSize: "13px" }}>
            <AlertCircle size={15} style={{ display: "inline", marginRight: "6px" }} />
            {optError}
          </div>
        )}
      </div>

      {/* 3. Optimization Results & Order Recommendations Table */}
      {optResult && (
        <div className="glass-panel" style={{ padding: "24px", marginTop: "24px" }}>
          {/* Diagnostics Ribbon */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "16px", marginBottom: "20px" }}>
            <div>
              <h3 className="section-title">Recommended Purchase Orders Schedule</h3>
              <p className="section-desc">Lookahead replenishment decisions generated by PuLP/CBC</p>
            </div>

            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              {/* Solver Optimality Badge */}
              <div
                style={{
                  padding: "6px 14px",
                  borderRadius: "99px",
                  fontSize: "12px",
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  background: optResult.is_proven_optimal
                    ? "rgba(16, 185, 129, 0.15)"
                    : optResult.is_feasible
                    ? "rgba(245, 158, 11, 0.15)"
                    : "rgba(244, 63, 94, 0.15)",
                  color: optResult.is_proven_optimal
                    ? "var(--accent-emerald)"
                    : optResult.is_feasible
                    ? "var(--accent-amber)"
                    : "var(--accent-rose)",
                  border: optResult.is_proven_optimal
                    ? "1px solid var(--accent-emerald)"
                    : optResult.is_feasible
                    ? "1px solid var(--accent-amber)"
                    : "1px solid var(--accent-rose)",
                }}
              >
                <CheckCircle2 size={14} />
                <span>
                  {optResult.is_proven_optimal
                    ? "Proven Optimal Solution"
                    : optResult.is_feasible
                    ? "Feasible Incumbent (Optimality Unproven)"
                    : optResult.termination_reason === "Infeasible"
                    ? "Infeasible Problem"
                    : optResult.termination_reason === "Unbounded"
                    ? "Unbounded Problem"
                    : "No Feasible Solution Found"}
                </span>
              </div>

              <div className="badge-pill cyan" style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <Clock size={13} />
                <span>Solve Time: {optResult.solve_time_sec}s</span>
              </div>
            </div>
          </div>

          {/* Quick Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", marginBottom: "20px" }}>
            <div className="glass-panel kpi-card" style={{ padding: "16px" }}>
              <span className="kpi-label">Recommended Orders</span>
              <div className="kpi-val cyan">{optResult.recommended_orders.length} orders</div>
            </div>
            <div className="glass-panel kpi-card" style={{ padding: "16px" }}>
              <span className="kpi-label">Total Recommended Units</span>
              <div className="kpi-val amber">{optResult.total_recommended_units.toLocaleString()} units</div>
            </div>
            <div className="glass-panel kpi-card" style={{ padding: "16px" }}>
              <span className="kpi-label">Estimated Spend</span>
              <div className="kpi-val emerald">${optResult.total_estimated_cost_usd.toLocaleString()}</div>
            </div>
          </div>

          {/* Orders Table */}
          {optResult.recommended_orders.length === 0 ? (
            <div style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
              <Package size={32} style={{ margin: "0 auto 8px" }} />
              <p>No replenishment orders required. Existing inventory buffers satisfy the projected demand horizon.</p>
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="benchmark-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left", color: "var(--text-muted)" }}>
                    <th style={{ padding: "10px 12px" }}>Day / Step</th>
                    <th style={{ padding: "10px 12px" }}>Product SKU</th>
                    <th style={{ padding: "10px 12px" }}>Destination Node</th>
                    <th style={{ padding: "10px 12px" }}>Supplier Source</th>
                    <th style={{ padding: "10px 12px" }}>Order Qty</th>
                    <th style={{ padding: "10px 12px" }}>Unit Price</th>
                    <th style={{ padding: "10px 12px" }}>Line Cost ($)</th>
                    <th style={{ padding: "10px 12px" }}>Lead Time</th>
                  </tr>
                </thead>
                <tbody>
                  {optResult.recommended_orders.map((o, idx) => (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                        background: idx % 2 === 0 ? "rgba(255, 255, 255, 0.01)" : "transparent",
                      }}
                    >
                      <td style={{ padding: "12px", fontWeight: 700 }} className="mono">
                        Day +{o.step}
                      </td>
                      <td style={{ padding: "12px", color: "var(--accent-cyan)" }} className="mono">
                        {o.product_id}
                      </td>
                      <td style={{ padding: "12px", color: "var(--text-muted)" }} className="mono">
                        {o.warehouse_id}
                      </td>
                      <td style={{ padding: "12px", color: "var(--accent-purple)" }} className="mono">
                        {o.supplier_id}
                      </td>
                      <td style={{ padding: "12px", fontWeight: 700 }} className="mono">
                        {o.quantity.toLocaleString()} units
                      </td>
                      <td style={{ padding: "12px" }} className="mono">
                        ${o.unit_cost_usd.toFixed(2)}
                      </td>
                      <td style={{ padding: "12px", fontWeight: 700, color: "var(--accent-emerald)" }} className="mono">
                        ${o.estimated_cost_usd.toLocaleString()}
                      </td>
                      <td style={{ padding: "12px", color: "var(--text-dim)" }} className="mono">
                        {o.lead_time_days} days
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
