import React from "react";
import { Truck, Factory, Warehouse as WhIcon, ShoppingCart, AlertTriangle } from "lucide-react";

export default function NetworkVisualizer({ currentStep, activeDisruptions = [], onHandUnits = 2800 }) {
  // Check disruption statuses
  const isPortCongested = activeDisruptions.some(d => d.type === "PORT_CONGESTION");
  const isSupplierDelayed = activeDisruptions.some(d => d.type === "SUPPLIER_DELAY");
  const isDemandSurge = activeDisruptions.some(d => d.type === "DEMAND_SPIKE");

  const suppliers = [
    { id: "SUP_01", name: "Apex Global", lead: "3d", rel: "98%", delayed: isSupplierDelayed },
    { id: "SUP_02", name: "PacRim Cargo", lead: "5d", rel: "94%", delayed: isPortCongested },
    { id: "SUP_03", name: "Metro Supply", lead: "2d", rel: "99%", delayed: false },
  ];

  const warehouses = [
    { id: "WH_01", name: "North Hub", cap: "40k", stockPct: 74 },
    { id: "WH_02", name: "Port DC", cap: "25k", stockPct: 58, warning: isPortCongested },
    { id: "WH_03", name: "Metro Hub", cap: "30k", stockPct: 82, surge: isDemandSurge },
  ];

  const demands = [
    { region: "Enterprise Retail", vol: "3,200/wk", surge: isDemandSurge },
    { region: "E-Commerce Direct", vol: "5,400/wk", surge: isDemandSurge },
    { region: "Wholesale B2B", vol: "1,800/wk", surge: false },
  ];

  return (
    <div className="glass-panel network-visualizer-card">
      <div className="card-top-bar">
        <div>
          <h3 className="section-title">Digital Twin Multi-Echelon Supply Network</h3>
          <p className="section-desc">
            Live telemetry of inbound procurement flows, in-transit buffer pipelines, and facility fulfillment
          </p>
        </div>
        {activeDisruptions.length > 0 && (
          <div className="disruption-badge-active">
            <AlertTriangle size={15} />
            <span>{activeDisruptions.length} Disruption Active (Day {currentStep})</span>
          </div>
        )}
      </div>

      <div className="network-flow-container">
        {/* Suppliers Column */}
        <div className="network-column">
          <div className="column-title">
            <Factory size={16} /> Tier-1 Suppliers
          </div>
          <div className="nodes-stack">
            {suppliers.map(s => (
              <div key={s.id} className={`node-item supplier-node ${s.delayed ? "disrupted-border" : ""}`}>
                <div className="node-header">
                  <span className="node-id">{s.id}</span>
                  <span className="node-badge">{s.lead} lead</span>
                </div>
                <div className="node-name">{s.name}</div>
                <div className="node-meta">
                  <span>Reliability: {s.rel}</span>
                  {s.delayed && <span className="error-text">⚠️ 2.5x Delay</span>}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Transit Pipeline Animation */}
        <div className="transit-pipeline-zone">
          <div className="pipeline-title">
            <Truck size={16} className="truck-icon" /> In-Transit Pipelines
          </div>
          <div className="pipeline-tracks">
            <div className="pipeline-track">
              <div className="freight-packet packet-1">📦 PO-841</div>
            </div>
            <div className="pipeline-track">
              <div className="freight-packet packet-2">🚢 SEA-192</div>
            </div>
            <div className="pipeline-track">
              <div className="freight-packet packet-3">🚚 TRK-044</div>
            </div>
          </div>
          <div className="pipeline-status">
            <span>Stochastic Delay Model: Gaussian + Disruption Multiplier</span>
          </div>
        </div>

        {/* Warehouses Column */}
        <div className="network-column">
          <div className="column-title">
            <WhIcon size={16} /> Distribution Centers
          </div>
          <div className="nodes-stack">
            {warehouses.map(w => (
              <div key={w.id} className={`node-item wh-node ${w.warning ? "warning-border" : ""}`}>
                <div className="node-header">
                  <span className="node-id">{w.id}</span>
                  <span className="node-badge">{w.cap} cap</span>
                </div>
                <div className="node-name">{w.name}</div>
                <div className="wh-capacity-bar">
                  <div
                    className={`wh-fill ${w.stockPct < 60 ? "amber" : "cyan"}`}
                    style={{ width: `${w.stockPct}%` }}
                  />
                </div>
                <div className="wh-util-text">{w.stockPct}% Utilization</div>
              </div>
            ))}
          </div>
        </div>

        {/* Customer Demand Column */}
        <div className="network-column">
          <div className="column-title">
            <ShoppingCart size={16} /> Customer Demand
          </div>
          <div className="nodes-stack">
            {demands.map((d, i) => (
              <div key={i} className={`node-item demand-node ${d.surge ? "surge-border" : ""}`}>
                <div className="node-header">
                  <span className="node-id">CHNL-0{i+1}</span>
                  {d.surge && <span className="surge-tag">Viral Surge 🔥</span>}
                </div>
                <div className="node-name">{d.region}</div>
                <div className="node-meta">
                  <span>Velocity: {d.vol}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
