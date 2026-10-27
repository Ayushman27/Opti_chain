import React from "react";
import { Trophy, TrendingUp, BarChart2, CheckCircle2 } from "lucide-react";

export default function BenchmarkView({ benchmarkData }) {
  const policies = benchmarkData?.policies || [];

  return (
    <div className="benchmark-page">
      <div className="benchmark-header-banner glass-panel">
        <div className="banner-badge">
          <Trophy size={18} className="trophy-icon" />
          <span>Multi-Paradigm Algorithmic Benchmark</span>
        </div>
        <h2 className="banner-title">Operations Research vs ML vs Deep RL vs Heuristics</h2>
        <p className="banner-subtitle">
          Standardized, multi-seed comparative benchmark evaluated across identical non-stationary demand
          tensors, stochastic lead times, and black-swan disruptions.
        </p>
      </div>

      {/* Top 3 Policy Cards */}
      <div className="podium-grid">
        {policies.slice(0, 3).map((p, idx) => (
          <div key={p.Policy} className={`glass-panel podium-card rank-${idx + 1}`}>
            <div className="podium-rank-tag">
              #{idx + 1} {idx === 0 ? "🏆 Global Optimal" : idx === 1 ? "🥈 Runner-up" : "🥉 High Resiliency"}
            </div>
            <div className="podium-policy-name">{p.Policy}</div>
            <div className="podium-profit">
              ${(p["Mean Profit ($)"] / 1000000).toFixed(2)}M
              <span className="profit-label">Net Profit</span>
            </div>
            <div className="podium-stats-row">
              <div>
                <span className="stat-sub">Fill Rate</span>
                <span className="stat-bold cyan">{p["Fill Rate (%)"]}%</span>
              </div>
              <div>
                <span className="stat-sub">Total Cost</span>
                <span className="stat-bold amber">${(p["Mean Total Cost ($)"] / 1000000).toFixed(2)}M</span>
              </div>
              <div>
                <span className="stat-sub">Turnover</span>
                <span className="stat-bold emerald">{p["Inventory Turnover"]}x</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Complete Comparison Table */}
      <div className="glass-panel benchmark-table-card">
        <h3 className="section-title">Comparative Performance Matrix</h3>
        <div className="table-responsive">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Policy Algorithm</th>
                <th>Fill Rate (%)</th>
                <th>Alpha SL (%)</th>
                <th>Mean Total Cost</th>
                <th>Mean Revenue</th>
                <th>Mean Net Profit</th>
                <th>Holding Cost</th>
                <th>Stockout Penalty</th>
                <th>Turnover</th>
              </tr>
            </thead>
            <tbody>
              {policies.map((p, i) => (
                <tr key={p.Policy} className={i === 0 ? "top-row" : ""}>
                  <td className="policy-name-cell">
                    <span className="bullet"></span>
                    <strong>{p.Policy}</strong>
                  </td>
                  <td className="mono cyan">{p["Fill Rate (%)"]}%</td>
                  <td className="mono">{p["Service Level (%)"]}%</td>
                  <td className="mono amber">${Number(p["Mean Total Cost ($)"]).toLocaleString()}</td>
                  <td className="mono">${Number(p["Mean Revenue ($)"]).toLocaleString()}</td>
                  <td className="mono emerald profit-cell">
                    ${Number(p["Mean Profit ($)"]).toLocaleString()}
                  </td>
                  <td className="mono">${Number(p["Holding Cost ($)"]).toLocaleString()}</td>
                  <td className="mono rose">${Number(p["Stockout Cost ($)"]).toLocaleString()}</td>
                  <td className="mono">{p["Inventory Turnover"]}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
