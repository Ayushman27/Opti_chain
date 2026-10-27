import React from "react";
import { TrendingUp, DollarSign, PackageCheck, AlertOctagon } from "lucide-react";

export default function KpiMetricsGrid({ kpis, costBreakdown }) {
  const fillRate = kpis?.fill_rate_percent ?? 92.4;
  const netProfit = kpis?.net_profit ?? 3280000;
  const totalCost = kpis?.total_cost ?? 12511000;
  const unmetUnits = kpis?.total_unmet_demand ?? 1420;
  const holdingCost = costBreakdown?.holding_cost ?? 181949;
  const stockoutPenalty = costBreakdown?.stockout_penalty ?? 1216464;

  return (
    <div className="kpi-grid">
      {/* Fill Rate OTIF */}
      <div className="glass-panel kpi-card">
        <div className="kpi-card-header">
          <span className="kpi-title">Order Fill Rate (OTIF)</span>
          <div className="kpi-icon-badge cyan">
            <PackageCheck size={18} />
          </div>
        </div>
        <div className="kpi-value cyan">{fillRate}%</div>
        <div className="kpi-bar-track">
          <div
            className="kpi-bar-fill cyan"
            style={{ width: `${Math.min(100, fillRate)}%` }}
          />
        </div>
        <div className="kpi-footer">
          <span>Target: 95.0%</span>
          <span className={fillRate >= 95 ? "positive" : "warning"}>
            {fillRate >= 95 ? "✓ Target Met" : "Buffer Required"}
          </span>
        </div>
      </div>

      {/* Net Profit */}
      <div className="glass-panel kpi-card">
        <div className="kpi-card-header">
          <span className="kpi-title">Cumulative Net Profit</span>
          <div className="kpi-icon-badge emerald">
            <TrendingUp size={18} />
          </div>
        </div>
        <div className="kpi-value emerald">
          ${(netProfit / 1000000).toFixed(2)}M
        </div>
        <div className="kpi-subtitle">
          Margin: {totalCost > 0 ? ((netProfit / (netProfit + totalCost)) * 100).toFixed(1) : 0}% of revenue
        </div>
        <div className="kpi-footer">
          <span>COGS & Ops Deducted</span>
          <span className="positive">+12.4% vs Baseline</span>
        </div>
      </div>

      {/* Total Supply Chain Cost */}
      <div className="glass-panel kpi-card">
        <div className="kpi-card-header">
          <span className="kpi-title">Total Operational Cost</span>
          <div className="kpi-icon-badge amber">
            <DollarSign size={18} />
          </div>
        </div>
        <div className="kpi-value amber">
          ${(totalCost / 1000000).toFixed(2)}M
        </div>
        <div className="kpi-sub-breakdown">
          <span>Holding: ${(holdingCost / 1000).toFixed(0)}k</span>
          <span>•</span>
          <span>Stockout: ${(stockoutPenalty / 1000).toFixed(0)}k</span>
        </div>
        <div className="kpi-footer">
          <span>Holding/Penalty Ratio</span>
          <span>1 : {(stockoutPenalty / Math.max(1, holdingCost)).toFixed(1)}</span>
        </div>
      </div>

      {/* Unmet Customer Demand */}
      <div className="glass-panel kpi-card">
        <div className="kpi-card-header">
          <span className="kpi-title">Total Unmet Demand</span>
          <div className="kpi-icon-badge rose">
            <AlertOctagon size={18} />
          </div>
        </div>
        <div className="kpi-value rose">
          {unmetUnits.toLocaleString()} units
        </div>
        <div className="kpi-subtitle">
          {kpis?.stockout_incident_count ?? 0} disruption incidents
        </div>
        <div className="kpi-footer">
          <span>Alpha Service Level</span>
          <span className="mono">{kpis?.alpha_service_level_percent ?? 0}%</span>
        </div>
      </div>
    </div>
  );
}
