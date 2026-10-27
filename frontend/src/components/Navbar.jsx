import React from "react";
import { Activity, Cpu, Database, Brain, Sparkles, ShieldCheck } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, currentStep, episodeDays, isOnline }) {
  const tabs = [
    { id: "simulation", label: "Digital Twin Simulator", icon: Activity },
    { id: "benchmark", label: "Policy Benchmark Matrix", icon: Cpu },
    { id: "forecasting", label: "Demand ML & Lead Times", icon: Database },
    { id: "agent", label: "Agentic AI Copilot", icon: Brain },
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
              <Icon size={16} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="navbar-status">
        <div className={`status-pill ${isOnline ? "online" : "standalone"}`}>
          <span className="status-dot"></span>
          <span>{isOnline ? "API Engine Live" : "Simulator Ready"}</span>
        </div>
        <div className="step-pill">
          <span>Day {currentStep} / {episodeDays}</span>
        </div>
      </div>
    </header>
  );
}
