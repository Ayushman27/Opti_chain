import React, { useState, useEffect } from "react";
import {
  DollarSign,
  TrendingUp,
  Percent,
  CheckCircle2,
  AlertTriangle,
  Package,
  Layers,
  Database,
  RefreshCw,
  AlertCircle,
} from "lucide-react";
import { fetchDashboardSummary, fetchBenchmarks } from "../services/api";

export default function OverviewView() {
  const [summary, setSummary] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, benchRes] = await Promise.all([
        fetchDashboardSummary(),
        fetchBenchmarks(),
      ]);
      setSummary(sumRes);
      setBenchmarks(benchRes);
    } catch (err) {
      console.error("Overview data load error:", err);
      setError(err.message || "Failed to load dashboard metrics from backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="glass-panel state-card">
        <RefreshCw className="animate-spin icon-spin" size={32} />
        <h3 className="state-title">Loading Executive Dashboard Summary...</h3>
        <p className="state-desc">Connecting to FastAPI service and parsing validated benchmark artifacts.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel state-card error-card">
        <AlertCircle size={40} className="icon-error" />
        <h3 className="state-title">Unable to Load Dashboard Data</h3>
        <p className="state-desc">{error}</p>
        <button onClick={loadData} className="btn-playback btn-play" style={{ marginTop: "16px" }}>
          <RefreshCw size={14} /> Retry Connection
        </button>
      </div>
    );
  }

  if (!summary || !benchmarks) {
    return (
      <div className="glass-panel state-card">
        <Package size={36} />
        <h3 className="state-title">No Benchmark Data Available</h3>
        <p className="state-desc">The backend returned an empty response. Verify docs/optimization_benchmark_results.json exists.</p>
      </div>
    );
  }

  const policies = benchmarks.policies || [];
  const maxCost = Math.max(...policies.map((p) => p.total_cost_usd || 1));
  const maxProfit = Math.max(...policies.map((p) => p.net_profit_usd || 1));
  const maxInv = Math.max(...policies.map((p) => p.average_inventory_units || 1));

  return (
    <div className="overview-page">
      {/* Header Provenance Banner */}
      <div className="glass-panel benchmark-header-banner">
        <div className="banner-badge">
          <Database size={16} />
          <span>Validated Multi-Seed Benchmark Provenance</span>
        </div>
        <h2 className="banner-title">Autonomous Digital Twin Supply Chain Overview</h2>
        <p className="banner-subtitle">
          Data Source: <code>{summary.data_source}</code>
        </p>
        <div className="provenance-tags" style={{ display: "flex", gap: "10px", marginTop: "12px" }}>
          <span className="badge-pill cyan">Top Policy: {summary.optimal_policy}</span>
          <span className="badge-pill emerald">Cost Savings vs ROP: {summary.cost_savings_vs_rop_percent}%</span>
          <span className="badge-pill purple">Profit Uplift: +{summary.net_profit_uplift_percent}%</span>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="kpi-grid">
        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Total Supply Chain Cost</span>
            <DollarSign size={18} className="kpi-icon cyan" />
          </div>
          <div className="kpi-val cyan">${summary.total_cost.toLocaleString()}</div>
          <div className="kpi-subtext">
            <span>vs ROP Baseline savings: </span>
            <span className="kpi-badge positive">-{summary.cost_savings_vs_rop_percent}%</span>
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Net Operating Profit</span>
            <TrendingUp size={18} className="kpi-icon emerald" />
          </div>
          <div className="kpi-val emerald">${summary.net_profit.toLocaleString()}</div>
          <div className="kpi-subtext">
            <span>Profit uplift vs ROP: </span>
            <span className="kpi-badge positive">+{summary.net_profit_uplift_percent}%</span>
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Customer Fill Rate</span>
            <Percent size={18} className="kpi-icon purple" />
          </div>
          <div className="kpi-val purple">{summary.fill_rate_percent}%</div>
          <div className="kpi-subtext">
            <span>Demand units fulfilled / total demand</span>
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Cycle Service Level</span>
            <CheckCircle2 size={18} className="kpi-icon amber" />
          </div>
          <div className="kpi-val amber">{summary.service_level_percent}%</div>
          <div className="kpi-subtext">
            <span>Zero-stockout order fulfillment cycles</span>
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Unmet Stockout Units</span>
            <AlertTriangle size={18} className="kpi-icon rose" />
          </div>
          <div className="kpi-val rose">{summary.stockout_units.toLocaleString()} units</div>
          <div className="kpi-subtext">
            <span>Mean lost units across 90-day horizon</span>
          </div>
        </div>

        <div className="glass-panel kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Average On-Hand Stock</span>
            <Package size={18} className="kpi-icon blue" />
          </div>
          <div className="kpi-val blue">{summary.average_inventory_units.toLocaleString()} units</div>
          <div className="kpi-subtext">
            <span>Annualized Turnover: </span>
            <span className="mono bold">{summary.inventory_turnover}x</span>
          </div>
        </div>
      </div>

      {/* Policy Comparison Charts Grid */}
      <div className="analytics-two-col-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginTop: "24px" }}>
        {/* Chart 1: Total Cost vs Net Profit */}
        <div className="glass-panel chart-card">
          <div className="chart-header">
            <div>
              <h3 className="section-title">Policy Financial Comparison</h3>
              <p className="section-desc">Total supply chain cost vs realized net profit across policies</p>
            </div>
          </div>

          <div className="policy-bars-list" style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "16px" }}>
            {policies.map((p, idx) => {
              const costPct = (p.total_cost_usd / maxCost) * 100;
              const profitPct = (p.net_profit_usd / maxProfit) * 100;
              const isMILP = p.policy.includes("MILP");
              return (
                <div key={idx} className="policy-bar-group" style={{ padding: "10px", borderRadius: "8px", background: isMILP ? "rgba(0, 242, 254, 0.05)" : "transparent", border: isMILP ? "1px solid rgba(0, 242, 254, 0.2)" : "1px solid transparent" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <span style={{ fontWeight: 700, fontSize: "13px", color: isMILP ? "var(--accent-cyan)" : "#fff" }}>
                      {p.policy} {isMILP && "🏆"}
                    </span>
                    <span className="mono" style={{ fontSize: "12px", color: "var(--accent-emerald)" }}>
                      Profit: ${p.net_profit_usd.toLocaleString()}
                    </span>
                  </div>

                  {/* Cost Bar */}
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", marginBottom: "4px" }}>
                    <span style={{ width: "45px", color: "var(--text-dim)" }}>Cost:</span>
                    <div style={{ flex: 1, height: "8px", background: "rgba(255, 255, 255, 0.08)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${costPct}%`, height: "100%", background: isMILP ? "var(--accent-cyan)" : "var(--accent-rose)", borderRadius: "4px" }} />
                    </div>
                    <span className="mono" style={{ width: "75px", textAlign: "right" }}>${p.total_cost_usd.toLocaleString()}</span>
                  </div>

                  {/* Profit Bar */}
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px" }}>
                    <span style={{ width: "45px", color: "var(--text-dim)" }}>Profit:</span>
                    <div style={{ flex: 1, height: "8px", background: "rgba(255, 255, 255, 0.08)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${profitPct}%`, height: "100%", background: "var(--accent-emerald)", borderRadius: "4px" }} />
                    </div>
                    <span className="mono" style={{ width: "75px", textAlign: "right" }}>${p.net_profit_usd.toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Chart 2: Inventory vs Service Level Trade-off */}
        <div className="glass-panel chart-card">
          <div className="chart-header">
            <div>
              <h3 className="section-title">Inventory vs Service Level Dynamics</h3>
              <p className="section-desc">Capital efficiency: Average inventory held vs Fill Rate & Service Level</p>
            </div>
          </div>

          <div className="policy-bars-list" style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "16px" }}>
            {policies.map((p, idx) => {
              const invPct = (p.average_inventory_units / maxInv) * 100;
              const isMILP = p.policy.includes("MILP");
              return (
                <div key={idx} className="policy-bar-group" style={{ padding: "10px", borderRadius: "8px", background: isMILP ? "rgba(0, 242, 254, 0.05)" : "transparent", border: isMILP ? "1px solid rgba(0, 242, 254, 0.2)" : "1px solid transparent" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <span style={{ fontWeight: 700, fontSize: "13px", color: isMILP ? "var(--accent-cyan)" : "#fff" }}>
                      {p.policy}
                    </span>
                    <span className="mono" style={{ fontSize: "12px", color: "var(--accent-purple)" }}>
                      Fill Rate: {p.fill_rate_percent}%
                    </span>
                  </div>

                  {/* Inventory Bar */}
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", marginBottom: "4px" }}>
                    <span style={{ width: "55px", color: "var(--text-dim)" }}>Avg Stock:</span>
                    <div style={{ flex: 1, height: "8px", background: "rgba(255, 255, 255, 0.08)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${invPct}%`, height: "100%", background: "var(--accent-blue)", borderRadius: "4px" }} />
                    </div>
                    <span className="mono" style={{ width: "75px", textAlign: "right" }}>{p.average_inventory_units.toFixed(1)} u</span>
                  </div>

                  {/* Service Level Indicator */}
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px" }}>
                    <span style={{ width: "55px", color: "var(--text-dim)" }}>Service:</span>
                    <div style={{ flex: 1, height: "8px", background: "rgba(255, 255, 255, 0.08)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${p.service_level_percent}%`, height: "100%", background: "var(--accent-amber)", borderRadius: "4px" }} />
                    </div>
                    <span className="mono" style={{ width: "75px", textAlign: "right" }}>{p.service_level_percent}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
