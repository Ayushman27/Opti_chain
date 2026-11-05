import React, { useState } from "react";
import {
  Truck,
  Eye,
  Sliders,
  ShieldCheck,
  Clock,
  ShoppingCart,
  IndianRupee,
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  PackageCheck,
  CheckCircle2,
  XCircle,
  Info,
  Layers,
  Maximize2,
  List,
} from "lucide-react";

// Reusable fixed-height empty placeholder to guarantee zero layout jumping
function StepFixedEmptyState({
  icon: Icon = Info,
  title = "No Events Recorded",
  description = "No activities scheduled for this step today.",
  height = 150,
}) {
  return (
    <div className="step-fixed-empty-state" style={{ height: `${height}px` }}>
      <Icon size={28} className="empty-icon" />
      <span className="empty-title">{title}</span>
      <span className="empty-desc">{description}</span>
    </div>
  );
}

export default function DailyStepInspector({
  currentStepData = {},
  currentStep = 0,
  totalSteps = 90,
  policyName = "Inventory Policy",
}) {
  const [activeTab, setActiveTab] = useState("pipeline"); // 'pipeline' | 'lifecycle' | 'sku_ledger'
  const [lifecycleViewMode, setLifecycleViewMode] = useState("all"); // 'all' | 'focus'
  const [focusedStepNum, setFocusedStepNum] = useState(6);

  const [expandedSteps, setExpandedSteps] = useState({
    6: true,
    7: true,
    8: true,
    9: true,
    10: true,
    11: true,
    12: true,
    13: true,
  });

  const toggleStep = (stepNum) => {
    setExpandedSteps((prev) => ({
      ...prev,
      [stepNum]: !prev[stepNum],
    }));
  };

  const executionSteps = currentStepData.execution_steps || [];
  const arrivedShipments = currentStepData.arrived_shipments || [];
  const placedOrders = currentStepData.placed_orders || [];
  const rejectedOrders = currentStepData.rejected_orders || [];
  const skuBreakdown = currentStepData.sku_breakdown || [];
  const activeDisruptions = currentStepData.active_disruptions || [];

  const getStepIcon = (stepNum) => {
    switch (stepNum) {
      case 6:
        return <PackageCheck size={18} />;
      case 7:
        return <Eye size={18} />;
      case 8:
        return <Sliders size={18} />;
      case 9:
        return <ShieldCheck size={18} />;
      case 10:
        return <Clock size={18} />;
      case 11:
        return <ShoppingCart size={18} />;
      case 12:
        return <IndianRupee size={18} />;
      case 13:
        return <Activity size={18} />;
      default:
        return <Activity size={18} />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "success":
        return "step-status-success";
      case "warning":
        return "step-status-warning";
      case "error":
        return "step-status-error";
      case "info":
        return "step-status-info";
      default:
        return "step-status-neutral";
    }
  };

  return (
    <div className="glass-panel daily-inspector-card fixed-layout-card" style={{ marginTop: "24px" }}>
      {/* Top Header Bar */}
      <div className="inspector-header">
        <div className="inspector-header-left">
          <div className="inspector-title-row">
            <span className="step-badge">Section 3.1 Mechanics</span>
            <h3 className="inspector-title">Daily Simulation Step Inspector</h3>
            <span className="day-counter-pill">
              Day <strong>{currentStep}</strong> of {totalSteps}
            </span>
          </div>
          <p className="inspector-desc">
            Chronological inspection of the 8 discrete-time events executed by the Digital Twin
            engine on Day {currentStep} under <strong>{policyName}</strong>.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="inspector-tabs">
          <button
            onClick={() => setActiveTab("pipeline")}
            className={`inspector-tab-btn ${activeTab === "pipeline" ? "active" : ""}`}
          >
            Inbound & Replenishment Pipeline ({placedOrders.length + arrivedShipments.length})
          </button>
          <button
            onClick={() => setActiveTab("lifecycle")}
            className={`inspector-tab-btn ${activeTab === "lifecycle" ? "active" : ""}`}
          >
            8-Step Execution Sequence
          </button>
          <button
            onClick={() => setActiveTab("sku_ledger")}
            className={`inspector-tab-btn ${activeTab === "sku_ledger" ? "active" : ""}`}
          >
            SKU Inventory Ledger ({skuBreakdown.length})
          </button>
        </div>
      </div>

      {/* Quick Daily Telemetry Ribbons - Fixed 7-Column Layout */}
      <div className="daily-stats-ribbon-fixed">
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Inbound Arrived</span>
          <span className="ribbon-val text-emerald">
            +{currentStepData.arrived_units || 0} units
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Orders Placed</span>
          <span className="ribbon-val text-cyan">
            {currentStepData.orders_placed_count || 0} POs
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Customer Demand</span>
          <span className="ribbon-val text-amber">
            {currentStepData.demand || 0} units
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Units Fulfilled</span>
          <span className="ribbon-val text-emerald">
            {currentStepData.fulfilled || 0} units
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Unmet Stockouts</span>
          <span className={`ribbon-val ${currentStepData.unmet > 0 ? "text-rose" : "text-muted"}`}>
            {currentStepData.unmet || 0} units
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Ending On-Hand</span>
          <span className="ribbon-val text-white">
            {Math.round(currentStepData.on_hand_units || 0)} units
          </span>
        </div>
        <div className="ribbon-stat-item-fixed">
          <span className="ribbon-label">Net Daily Profit</span>
          <span className={`ribbon-val ${(currentStepData.net_profit || 0) >= 0 ? "text-emerald" : "text-rose"}`}>
            {(currentStepData.net_profit || 0) < 0
              ? `-₹${Math.abs(Number(currentStepData.net_profit || 0)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : `₹${Number(currentStepData.net_profit || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          </span>
        </div>
      </div>

      {/* TAB 1: Inbound Deliveries & Replenishment Pipeline (FIXED BIG BOXES) */}
      {activeTab === "pipeline" && (
        <div className="inspector-fixed-subtab-container">
          <div className="pipeline-sections-fixed-grid">
            {/* Box 1: Inbound Deliveries */}
            <div className="pipeline-fixed-card">
              <div className="pipeline-card-top">
                <h4 className="pipeline-fixed-title text-emerald">
                  <PackageCheck size={18} /> Inbound Shipments Delivered Today (Day {currentStep})
                </h4>
                <span className="pipeline-count-pill text-emerald">
                  {arrivedShipments.length} Arrived ({currentStepData.arrived_units || 0} units)
                </span>
              </div>

              <div className="pipeline-card-body-fixed">
                {arrivedShipments.length > 0 ? (
                  <div className="step-fixed-table-wrap">
                    <table className="step-mini-table">
                      <thead>
                        <tr>
                          <th>Order ID</th>
                          <th>Product</th>
                          <th>Supplier</th>
                          <th>Warehouse</th>
                          <th>Delivered Quantity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {arrivedShipments.map((a, i) => (
                          <tr key={i}>
                            <td><code className="code-badge">{a.order_id}</code></td>
                            <td><strong>{a.product_name}</strong></td>
                            <td>{a.supplier_name}</td>
                            <td>{a.warehouse_id}</td>
                            <td className="text-emerald font-bold">+{a.quantity} units</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <StepFixedEmptyState
                    icon={PackageCheck}
                    title="No Inbound Deliveries Arriving Today"
                    description={`No shipments scheduled for delivery on Day ${currentStep}. Outstanding orders remain in-transit.`}
                    height={220}
                  />
                )}
              </div>
            </div>

            {/* Box 2: Newly Placed Replenishment Orders */}
            <div className="pipeline-fixed-card">
              <div className="pipeline-card-top">
                <h4 className="pipeline-fixed-title text-cyan">
                  <Truck size={18} /> Replenishment Purchase Orders Placed Today (Day {currentStep})
                </h4>
                <span className="pipeline-count-pill text-cyan">
                  {placedOrders.length} Dispatched
                </span>
              </div>

              <div className="pipeline-card-body-fixed">
                {placedOrders.length > 0 ? (
                  <div className="step-fixed-table-wrap">
                    <table className="step-mini-table">
                      <thead>
                        <tr>
                          <th>PO Reference</th>
                          <th>Product</th>
                          <th>Supplier</th>
                          <th>Qty Placed</th>
                          <th>Lead Time</th>
                          <th>Arrival Day</th>
                        </tr>
                      </thead>
                      <tbody>
                        {placedOrders.map((o, i) => (
                          <tr key={i}>
                            <td><code className="code-badge">{o.order_id}</code></td>
                            <td><strong>{o.product_name}</strong></td>
                            <td>{o.supplier_name}</td>
                            <td className="text-cyan font-bold">{o.final_qty} units</td>
                            <td className="text-indigo font-bold">{o.lead_time_days} days</td>
                            <td className="text-emerald font-bold">Day {o.expected_arrival_day}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <StepFixedEmptyState
                    icon={Truck}
                    title="Zero Purchase Orders Placed Today"
                    description={`Inventory Position (IP = ${Math.round(currentStepData.on_hand_units || 0)}) remains above the reorder threshold.`}
                    height={220}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: 8-Step Lifecycle Execution (FIXED HEIGHT STEPS) */}
      {activeTab === "lifecycle" && (
        <div className="execution-lifecycle-container">
          {/* View mode toggle (All Stacked vs Focused Stage) */}
          <div className="lifecycle-toolbar">
            <span className="toolbar-label">Execution View Mode:</span>
            <div className="toolbar-btn-group">
              <button
                onClick={() => setLifecycleViewMode("all")}
                className={`toolbar-btn ${lifecycleViewMode === "all" ? "active" : ""}`}
              >
                <List size={14} /> All 8 Steps (Fixed Stack)
              </button>
              <button
                onClick={() => setLifecycleViewMode("focus")}
                className={`toolbar-btn ${lifecycleViewMode === "focus" ? "active" : ""}`}
              >
                <Maximize2 size={14} /> Focused Step Stage
              </button>
            </div>
          </div>

          {/* If Focused View: Step Selector Pills */}
          {lifecycleViewMode === "focus" && (
            <div className="focus-step-selector-row">
              {executionSteps.map((s) => (
                <button
                  key={s.step_num}
                  onClick={() => setFocusedStepNum(s.step_num)}
                  className={`focus-selector-chip ${focusedStepNum === s.step_num ? "active" : ""}`}
                >
                  <span className="chip-step-num">Step {s.step_num}</span>
                  <span className="chip-step-tag">{s.tag}</span>
                </button>
              ))}
            </div>
          )}

          {/* Steps List */}
          <div className="execution-steps-stream">
            {executionSteps
              .filter((s) => (lifecycleViewMode === "focus" ? s.step_num === focusedStepNum : true))
              .map((step) => {
                const isExpanded = !!expandedSteps[step.step_num];
                const statusClass = getStatusColor(step.status);

                return (
                  <div key={step.step_num} className={`step-card fixed-step-card ${statusClass}`}>
                    <div className="step-card-header fixed-card-header" onClick={() => toggleStep(step.step_num)}>
                      <div className="step-card-header-left">
                        <span className="step-num-pill">Step {step.step_num}</span>
                        <div className="step-icon-wrapper">{getStepIcon(step.step_num)}</div>
                        <div className="step-card-title-group">
                          <span className="step-card-title">{step.title}</span>
                          <span className="step-card-tag">{step.tag}</span>
                        </div>
                      </div>

                      <div className="step-card-header-right">
                        <p className="step-card-summary text-truncate" title={step.summary}>
                          {step.summary}
                        </p>
                        <button className="btn-toggle-step" aria-label="Toggle Step Details">
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>
                      </div>
                    </div>

                    {/* Step Body with Fixed Height */}
                    {isExpanded && (
                      <div className="step-card-body fixed-card-body">
                        {/* Step 6: Arrivals (Fixed 160px) */}
                        {step.step_num === 6 && (
                          <div className="step-fixed-inner-wrap" style={{ height: "160px" }}>
                            {arrivedShipments.length > 0 ? (
                              <div className="step-fixed-table-wrap" style={{ height: "100%" }}>
                                <table className="step-mini-table">
                                  <thead>
                                    <tr>
                                      <th>Order ID</th>
                                      <th>Product / SKU</th>
                                      <th>Supplier</th>
                                      <th>Warehouse</th>
                                      <th>Quantity Received</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {arrivedShipments.map((a, i) => (
                                      <tr key={i}>
                                        <td><code className="code-badge">{a.order_id}</code></td>
                                        <td><strong>{a.product_name}</strong> <span className="sub-id">({a.product_id})</span></td>
                                        <td>{a.supplier_name}</td>
                                        <td>{a.warehouse_id}</td>
                                        <td className="text-emerald font-bold">+{a.quantity} units</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <StepFixedEmptyState
                                icon={PackageCheck}
                                title={`No Inbound Deliveries Arriving on Day ${currentStep}`}
                                description="In-transit shipments remain en route according to their sampled delivery lead times."
                                height={160}
                              />
                            )}
                          </div>
                        )}

                        {/* Step 7: Observation (Fixed 160px) */}
                        {step.step_num === 7 && (
                          <div className="step-observation-box" style={{ height: "160px", overflowY: "auto" }}>
                            <div className="observation-grid">
                              <div className="obs-item">
                                <span className="obs-label">Physical On-Hand Inventory:</span>
                                <span className="obs-value text-cyan">
                                  {step.data?.on_hand?.toLocaleString()} units
                                </span>
                              </div>
                              <div className="obs-item">
                                <span className="obs-label">In-Transit Pipeline:</span>
                                <span className="obs-value text-indigo">
                                  {step.data?.pipeline?.toLocaleString()} units
                                </span>
                              </div>
                              <div className="obs-item">
                                <span className="obs-label">Effective Inventory Position (IP):</span>
                                <span className="obs-value text-emerald font-bold">
                                  {step.data?.effective_inventory?.toLocaleString()} units
                                </span>
                              </div>
                            </div>

                            {activeDisruptions.length > 0 ? (
                              <div className="disruption-alert-card">
                                <AlertTriangle size={18} className="text-rose" />
                                <div className="disruption-alert-content">
                                  <div className="disruption-alert-title">
                                    Active Disruption Event on Day {currentStep}:
                                  </div>
                                  {activeDisruptions.map((d, idx) => (
                                    <div key={idx} className="disruption-alert-desc">
                                      <strong>{d.type}:</strong> {d.description} (Severity Multiplier: {d.severity}x)
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : (
                              <div className="disruption-clean-card">
                                <CheckCircle2 size={16} className="text-emerald" />
                                <span>Supply network operating normally. No active disruptions impacting suppliers or ports today.</span>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Step 8: Policy Decisions (Fixed 160px) */}
                        {step.step_num === 8 && (
                          <div className="step-fixed-inner-wrap" style={{ height: "160px" }}>
                            {placedOrders.length > 0 ? (
                              <div className="step-fixed-table-wrap" style={{ height: "100%" }}>
                                <table className="step-mini-table">
                                  <thead>
                                    <tr>
                                      <th>Target SKU</th>
                                      <th>Supplier</th>
                                      <th>Requested Qty</th>
                                      <th>Final Order Qty</th>
                                      <th>Unit Cost</th>
                                      <th>Estimated Line Total</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {placedOrders.map((o, idx) => (
                                      <tr key={idx}>
                                        <td><strong>{o.product_name}</strong> <span className="sub-id">({o.product_id})</span></td>
                                        <td>{o.supplier_name}</td>
                                        <td>{o.requested_qty} units</td>
                                        <td className="text-cyan font-bold">{o.final_qty} units</td>
                                        <td>₹{o.unit_cost?.toFixed(2)}</td>
                                        <td className="font-bold">₹{o.line_procurement_cost?.toFixed(2)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <StepFixedEmptyState
                                icon={Sliders}
                                title="No Purchase Orders Requested by Policy"
                                description={`Inventory Position (IP = ${Math.round(currentStepData.on_hand_units || 0)}) was evaluated above the reorder point.`}
                                height={160}
                              />
                            )}
                          </div>
                        )}

                        {/* Step 9: Supplier & MOQ Constraints (Fixed 160px) */}
                        {step.step_num === 9 && (
                          <div className="step-fixed-inner-wrap" style={{ height: "160px" }}>
                            {placedOrders.length > 0 ? (
                              <div className="step-fixed-table-wrap" style={{ height: "100%" }}>
                                <table className="step-mini-table">
                                  <thead>
                                    <tr>
                                      <th>Target SKU</th>
                                      <th>Supplier</th>
                                      <th>Requested</th>
                                      <th>Supplier MOQ</th>
                                      <th>Daily Capacity</th>
                                      <th>Validation Status</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {placedOrders.map((o, idx) => (
                                      <tr key={idx}>
                                        <td><strong>{o.product_name}</strong></td>
                                        <td>{o.supplier_name}</td>
                                        <td>{o.requested_qty} units</td>
                                        <td>{o.moq} units</td>
                                        <td>{o.daily_capacity} cap</td>
                                        <td>
                                          {o.moq_applied ? (
                                            <span className="constraint-badge warning">
                                              MOQ Enforced ({o.requested_qty} → {o.final_qty})
                                            </span>
                                          ) : (
                                            <span className="constraint-badge success">
                                              MOQ & Capacity Satisfied
                                            </span>
                                          )}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <StepFixedEmptyState
                                icon={ShieldCheck}
                                title="All Supplier Constraints Verified"
                                description="No purchase orders required MOQ clipping or supplier daily capacity enforcement today."
                                height={160}
                              />
                            )}
                          </div>
                        )}

                        {/* Step 10: Stochastic Lead Time (Fixed 160px) */}
                        {step.step_num === 10 && (
                          <div className="step-fixed-inner-wrap" style={{ height: "160px" }}>
                            {step.items && step.items.length > 0 ? (
                              <div className="step-fixed-table-wrap" style={{ height: "100%" }}>
                                <table className="step-mini-table">
                                  <thead>
                                    <tr>
                                      <th>PO Reference</th>
                                      <th>Product</th>
                                      <th>Supplier</th>
                                      <th>Order Quantity</th>
                                      <th>Sampled Lead Time</th>
                                      <th>Scheduled Delivery Day</th>
                                      <th>Physical Rule</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {step.items.map((it, idx) => (
                                      <tr key={idx}>
                                        <td><code className="code-badge">{it.order_id}</code></td>
                                        <td><strong>{it.sku}</strong></td>
                                        <td>{it.supplier}</td>
                                        <td className="font-bold">{it.quantity} units</td>
                                        <td className="text-indigo font-bold">{it.lead_time}</td>
                                        <td className="text-emerald font-bold">{it.arrival_day}</td>
                                        <td>
                                          <span className="constraint-badge success">
                                            L ≥ 1 day enforced (No instant arrival)
                                          </span>
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <StepFixedEmptyState
                                icon={Clock}
                                title="No New Pipeline Dispatches Today"
                                description="No new orders were placed. Current in-transit shipments continue moving toward delivery."
                                height={160}
                              />
                            )}
                          </div>
                        )}

                        {/* Step 11: Demand Realization & Fulfillment (Fixed 220px) */}
                        {step.step_num === 11 && (
                          <div className="step-fixed-inner-wrap" style={{ height: "220px" }}>
                            <div className="step-fixed-table-wrap" style={{ height: "100%" }}>
                              <table className="step-mini-table">
                                <thead>
                                  <tr>
                                    <th>Product</th>
                                    <th>Realized Demand</th>
                                    <th>Fulfilled Units</th>
                                    <th>Stockout / Unmet</th>
                                    <th>Fulfillment Status</th>
                                    <th>Remaining Stock</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {(step.sku_breakdown || []).map((sku, idx) => (
                                    <tr key={idx}>
                                      <td><strong>{sku.product_name}</strong> <span className="sub-id">({sku.category})</span></td>
                                      <td>{sku.demand} units</td>
                                      <td className="text-emerald font-bold">{sku.fulfilled} units</td>
                                      <td className={sku.unmet > 0 ? "text-rose font-bold" : "text-muted"}>
                                        {sku.unmet} units
                                      </td>
                                      <td>
                                        {sku.unmet > 0 ? (
                                          <span className="constraint-badge error">
                                            Stockout Occurred
                                          </span>
                                        ) : (
                                          <span className="constraint-badge success">
                                            100% Demand Fulfilled
                                          </span>
                                        )}
                                      </td>
                                      <td className="font-bold">{sku.ending_on_hand} units</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Step 12: Financial Accounting (Fixed 150px) */}
                        {step.step_num === 12 && step.breakdown && (
                          <div className="financials-matrix-grid" style={{ minHeight: "150px" }}>
                            <div className="fin-card">
                              <span className="fin-label">Daily Revenue</span>
                              <span className="fin-val text-emerald">
                                +₹{step.breakdown.revenue?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card">
                              <span className="fin-label">Holding Cost</span>
                              <span className="fin-val text-rose">
                                -₹{step.breakdown.holding_cost?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card">
                              <span className="fin-label">Stockout Penalties</span>
                              <span className={`fin-val ${step.breakdown.stockout_penalty > 0 ? "text-rose" : "text-muted"}`}>
                                -₹{step.breakdown.stockout_penalty?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card">
                              <span className="fin-label">PO Setup Fixed Cost</span>
                              <span className="fin-val text-rose">
                                -₹{step.breakdown.fixed_order_cost?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card">
                              <span className="fin-label">Procurement Cost</span>
                              <span className="fin-val text-rose">
                                -₹{step.breakdown.procurement_cost?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card">
                              <span className="fin-label">Transportation</span>
                              <span className="fin-val text-rose">
                                -₹{step.breakdown.transport_cost?.toFixed(2)}
                              </span>
                            </div>
                            <div className="fin-card fin-net">
                              <span className="fin-label">Net Daily Profit</span>
                              <span className={`fin-val ${step.breakdown.net_profit >= 0 ? "text-emerald" : "text-rose"}`}>
                                {step.breakdown.net_profit < 0 ? `-₹${Math.abs(step.breakdown.net_profit).toFixed(2)}` : `₹${step.breakdown.net_profit?.toFixed(2)}`}
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Step 13: Telemetry (Fixed 120px) */}
                        {step.step_num === 13 && step.data && (
                          <div className="telemetry-summary-box" style={{ minHeight: "120px" }}>
                            <div className="tele-metric">
                              <span className="tele-label">Physical Ending Inventory:</span>
                              <span className="tele-val text-white">{step.data.ending_on_hand_units} units</span>
                            </div>
                            <div className="tele-metric">
                              <span className="tele-label">Asset Balance-Sheet Valuation:</span>
                              <span className="tele-val text-emerald">₹{step.data.ending_on_hand_val?.toLocaleString()}</span>
                            </div>
                            <div className="tele-metric">
                              <span className="tele-label">Pipeline Buffer Stock:</span>
                              <span className="tele-val text-indigo">{step.data.pipeline_units} units in-transit</span>
                            </div>
                            <div className="tele-metric">
                              <span className="tele-label">Daily Realized Fill Rate:</span>
                              <span className="tele-val text-cyan">{step.data.fill_rate}%</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* TAB 3: SKU Inventory Ledger (FIXED BIG TABLE) */}
      {activeTab === "sku_ledger" && (
        <div className="inspector-fixed-subtab-container">
          <div className="step-fixed-table-wrap" style={{ height: "340px" }}>
            <table className="step-mini-table full-ledger-table">
              <thead>
                <tr>
                  <th>Product SKU</th>
                  <th>Category</th>
                  <th>Unit Cost</th>
                  <th>Selling Price</th>
                  <th>Realized Demand</th>
                  <th>Fulfilled Units</th>
                  <th>Stockout / Unmet</th>
                  <th>Ending Physical Stock</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {skuBreakdown.map((s, idx) => (
                  <tr key={idx}>
                    <td><strong>{s.product_name}</strong> <span className="sub-id">({s.product_id})</span></td>
                    <td><span className="category-pill">{s.category}</span></td>
                    <td>₹{s.unit_cost?.toFixed(2)}</td>
                    <td>₹{s.selling_price?.toFixed(2)}</td>
                    <td className="font-bold">{s.demand}</td>
                    <td className="text-emerald font-bold">{s.fulfilled}</td>
                    <td className={s.unmet > 0 ? "text-rose font-bold" : "text-muted"}>
                      {s.unmet}
                    </td>
                    <td className="text-cyan font-bold">{s.ending_on_hand} units</td>
                    <td>
                      {s.unmet > 0 ? (
                        <span className="constraint-badge error">Shortage Stockout</span>
                      ) : s.ending_on_hand < 50 ? (
                        <span className="constraint-badge warning">Low Stock</span>
                      ) : (
                        <span className="constraint-badge success">Healthy Stock</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
