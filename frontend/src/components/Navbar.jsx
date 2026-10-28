import React from "react";
import {
  LayoutDashboard,
  LineChart,
  Package,
  Cpu,
  Activity,
  Brain,
  Sparkles,
} from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, currentStep, episodeDays, isOnline }) {
  const tabs = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "forecasting", label: "Forecasting", icon: LineChart },
    { id: "inventory", label: "Inventory", icon: Package },
    { id: "optimization", label: "Optimization", icon: Cpu },
    { id: "simulation", label: "Digital Twin", icon: Activity },
    { id: "agent", label: "AI Copilot", icon: Brain },
  ];

  return (
    <header className="navbar-container">
      <div className="navbar-brand">
        <div className="brand-icon-wrapper">
          <Sparkles size={20} className="brand-sparkle" />
        </div>
        <div>
          <div className="brand-title">
            OptiChain <span className="brand-ai">AI</span>
          </div>
          <div className="brand-subtitle">Autonomous Supply Chain Digital Twin</div>
        </div>
      </div>

      <nav className="navbar-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`nav-tab-btn ${isActive ? "active" : ""}`}
            >
              <Icon size={15} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="navbar-status">
        <div className={`status-pill ${isOnline ? "online" : "offline"}`}>
          <span className="status-dot"></span>
          <span>{isOnline ? "FastAPI Live" : "API Offline"}</span>
        </div>
        <div className="step-pill">
          <span>Day {currentStep} / {episodeDays}</span>
        </div>
      </div>
    </header>
  );
}
