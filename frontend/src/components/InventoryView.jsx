import React, { useState, useEffect } from "react";
import {
  Package,
  Truck,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  AlertCircle,
  Database,
  ArrowRight,
} from "lucide-react";
import { fetchInventory } from "../services/api";

export default function InventoryView() {
  const [invData, setInvData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadInventory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchInventory();
      setInvData(data);
    } catch (err) {
      console.error("Inventory fetch error:", err);
      setError(err.message || "Failed to load inventory status from backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInventory();
  }, []);

  if (loading) {
    return (
      <div className="glass-panel state-card">
        <RefreshCw className="animate-spin icon-spin" size={32} />
        <h3 className="state-title">Retrieving Live Inventory Snapshot...</h3>
        <p className="state-desc">Synchronizing SKU on-hand balances, in-transit pipeline orders, and fulfillment telemetry.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel state-card error-card">
        <AlertCircle size={40} className="icon-error" />
        <h3 className="state-title">Inventory Synchronization Failed</h3>
        <p className="state-desc">{error}</p>
        <button onClick={loadInventory} className="btn-playback btn-play" style={{ marginTop: "16px" }}>
          <RefreshCw size={14} /> Retry Snapshot
        </button>
      </div>
    );
  }

  if (!invData || !invData.products || invData.products.length === 0) {
    return (
      <div className="glass-panel state-card">
        <Package size={36} />
        <h3 className="state-title">No Inventory Records Found</h3>
        <p className="state-desc">Simulator environment currently reports zero active SKUs.</p>
      </div>
    );
  }

  const products = invData.products;

  const getStatusBadge = (status) => {
    switch (status) {
      case "CRITICAL_STOCKOUT":
        return <span className="badge-pill rose">Critical Stockout</span>;
      case "LOW_STOCK":
        return <span className="badge-pill amber">Low Stock</span>;
      case "OVERSTOCKED":
        return <span className="badge-pill purple">Overstocked</span>;
      default:
        return <span className="badge-pill emerald">Optimal Stock</span>;
    }
  };

  return (
    <div className="inventory-page">
      {/* Header Banner */}
      <div className="glass-panel benchmark-header-banner">
        <div className="banner-badge">
          <Database size={16} />
          <span>Multi-Echelon Physical Inventory & Replenishment Pipeline</span>
        </div>
        <h2 className="banner-title">SKU Inventory Positions & Order Recommendations</h2>
        <p className="banner-subtitle">
          Data Source: <code>{invData.data_source}</code> (As of Step/Day {invData.as_of_step})
        </p>

        {/* Global Summary Stats */}
        <div className="kpi-grid" style={{ marginTop: "20px" }}>
          <div className="glass-panel kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Total On-Hand Stock</span>
              <Package size={18} className="kpi-icon cyan" />
            </div>
            <div className="kpi-val cyan">{invData.total_on_hand_units.toLocaleString()} units</div>
            <div className="kpi-subtext">Physical warehouse inventory</div>
          </div>

          <div className="glass-panel kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">In-Transit Pipeline</span>
              <Truck size={18} className="kpi-icon amber" />
            </div>
            <div className="kpi-val amber">{invData.total_pipeline_units.toLocaleString()} units</div>
            <div className="kpi-subtext">Replenishment orders en route</div>
          </div>

          <div className="glass-panel kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Fulfilled Customer Demand</span>
              <CheckCircle2 size={18} className="kpi-icon emerald" />
            </div>
            <div className="kpi-val emerald">{invData.total_fulfilled_demand.toLocaleString()} units</div>
            <div className="kpi-subtext">Satisfied demand volume</div>
          </div>

          <div className="glass-panel kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Unmet Lost Demand</span>
              <AlertTriangle size={18} className="kpi-icon rose" />
            </div>
            <div className="kpi-val rose">{invData.total_unmet_demand.toLocaleString()} units</div>
            <div className="kpi-subtext">Lost sales from stockout</div>
          </div>
        </div>
      </div>

      {/* Product Inventory Table Card */}
      <div className="glass-panel table-card" style={{ marginTop: "24px", padding: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <div>
            <h3 className="section-title">Product Stock & Buffer Levels</h3>
            <p className="section-desc">Continuous monitoring of on-hand physical stock, ROP reorder thresholds, and safety buffers</p>
          </div>
          <button onClick={loadInventory} className="btn-step" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <RefreshCw size={13} /> Refresh
          </button>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table className="benchmark-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left", color: "var(--text-muted)" }}>
                <th style={{ padding: "12px 14px" }}>SKU / Product</th>
                <th style={{ padding: "12px 14px" }}>Category</th>
                <th style={{ padding: "12px 14px" }}>On-Hand</th>
                <th style={{ padding: "12px 14px" }}>In Pipeline</th>
                <th style={{ padding: "12px 14px" }}>Daily Demand</th>
                <th style={{ padding: "12px 14px" }}>Safety Stock</th>
                <th style={{ padding: "12px 14px" }}>Reorder Point</th>
                <th style={{ padding: "12px 14px" }}>Status</th>
                <th style={{ padding: "12px 14px" }}>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {products.map((p, idx) => {
                const totalPosition = p.on_hand_units + p.pipeline_units;
                return (
                  <tr
                    key={p.product_id}
                    style={{
                      borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                      background: idx % 2 === 0 ? "rgba(255, 255, 255, 0.01)" : "transparent",
                    }}
                  >
                    <td style={{ padding: "14px", fontWeight: 700 }}>
                      <div style={{ color: "#fff" }}>{p.product_name}</div>
                      <div className="mono" style={{ fontSize: "11px", color: "var(--text-dim)" }}>
                        {p.product_id} (${p.unit_cost.toFixed(2)} cost)
                      </div>
                    </td>
                    <td style={{ padding: "14px", color: "var(--text-muted)" }}>{p.category}</td>
                    <td style={{ padding: "14px" }}>
                      <span className="mono" style={{ fontWeight: 700, color: p.on_hand_units === 0 ? "var(--accent-rose)" : "var(--accent-cyan)" }}>
                        {p.on_hand_units.toFixed(1)}
                      </span>
                    </td>
                    <td style={{ padding: "14px" }}>
                      <span className="mono" style={{ color: p.pipeline_units > 0 ? "var(--accent-amber)" : "var(--text-dim)" }}>
                        {p.pipeline_units > 0 ? `+${p.pipeline_units.toFixed(1)}` : "—"}
                      </span>
                    </td>
                    <td style={{ padding: "14px" }} className="mono">
                      {p.recent_demand_rate.toFixed(1)} u/d
                    </td>
                    <td style={{ padding: "14px", color: "var(--text-dim)" }} className="mono">
                      {p.safety_stock.toFixed(1)}
                    </td>
                    <td style={{ padding: "14px" }}>
                      <span className="mono" style={{ fontWeight: 600, color: "var(--accent-purple)" }}>
                        {p.reorder_point.toFixed(1)}
                      </span>
                    </td>
                    <td style={{ padding: "14px" }}>
                      {getStatusBadge(p.stock_status)}
                    </td>
                    <td style={{ padding: "14px", fontSize: "12px", color: "var(--text-muted)" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <ArrowRight size={13} className="cyan" style={{ flexShrink: 0 }} />
                        <span>{p.recommendation || "Maintain current replenishment parameters."}</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
