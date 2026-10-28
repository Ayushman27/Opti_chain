import React, { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import OverviewView from "./components/OverviewView";
import ForecastView from "./components/ForecastView";
import InventoryView from "./components/InventoryView";
import OptimizationView from "./components/OptimizationView";
import KpiMetricsGrid from "./components/KpiMetricsGrid";
import SimulationControls from "./components/SimulationControls";
import NetworkVisualizer from "./components/NetworkVisualizer";
import InventoryChart from "./components/InventoryChart";
import AgentChat from "./components/AgentChat";
import { fetchHealth, executeSimulation } from "./services/api";
import { AlertCircle, RefreshCw } from "lucide-react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [isOnline, setIsOnline] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [healthError, setHealthError] = useState(null);

  // Digital Twin Simulation State
  const [policy, setPolicy] = useState("min_max");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [simLoading, setSimLoading] = useState(false);
  const [simData, setSimData] = useState(null);

  const timerRef = useRef(null);

  const checkApiHealth = async () => {
    setCheckingHealth(true);
    setHealthError(null);
    try {
      const res = await fetchHealth();
      setIsOnline(res.status === "healthy" || res.status === "degraded");
    } catch (err) {
      console.warn("API health check failed:", err);
      setIsOnline(false);
      setHealthError("FastAPI backend is offline or unreachable at http://localhost:8000.");
    } finally {
      setCheckingHealth(false);
    }
  };

  useEffect(() => {
    checkApiHealth();
  }, []);

  // Run simulation when simulation tab is active or requested
  const handleRunSimulation = async (selectedPolicy) => {
    setSimLoading(true);
    setIsPlaying(false);
    setCurrentStep(0);
    try {
      const res = await executeSimulation({
        policy: selectedPolicy || policy,
        episodeDays: 90,
        randomSeed: 42,
        nWarehouses: 1,
        nProducts: 5,
        nSuppliers: 3,
      });
      setSimData(res);
    } catch (err) {
      console.error("Simulation run error:", err);
    } finally {
      setSimLoading(false);
    }
  };

  // Playback timer for simulation step scrubber
  useEffect(() => {
    if (isPlaying && simData?.step_history_sample?.length) {
      const intervalMs = Math.max(100, Math.round(500 / speed));
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          const max = (simData.step_history_sample.length || 20) - 1;
          if (prev >= max) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, speed, simData]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const currentStepData = simData?.step_history_sample?.[currentStep] || {};

  return (
    <div className="app-wrapper">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentStep={currentStep}
        episodeDays={simData?.episode_days || 90}
        isOnline={isOnline}
      />

      {/* Offline Alert Banner if backend unreachable */}
      {!isOnline && !checkingHealth && (
        <div
          style={{
            background: "rgba(244, 63, 94, 0.12)",
            borderBottom: "1px solid rgba(244, 63, 94, 0.3)",
            padding: "10px 32px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "13px",
            color: "var(--accent-rose)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertCircle size={16} />
            <span>
              <strong>Backend Offline:</strong> {healthError || "FastAPI backend is unreachable."} Start the API using <code>uvicorn optichain.api.main:app --reload --port 8000</code>.
            </span>
          </div>
          <button
            onClick={checkApiHealth}
            style={{
              background: "rgba(244, 63, 94, 0.2)",
              border: "1px solid var(--accent-rose)",
              color: "#fff",
              padding: "4px 12px",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "12px",
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <RefreshCw size={12} /> Retry Ping
          </button>
        </div>
      )}

      <main className="main-content" style={{ padding: "24px 32px", maxWidth: "1400px", margin: "0 auto", width: "100%" }}>
        {activeTab === "overview" && <OverviewView />}

        {activeTab === "forecasting" && <ForecastView />}

        {activeTab === "inventory" && <InventoryView />}

        {activeTab === "optimization" && <OptimizationView />}

        {activeTab === "simulation" && (
          <div className="simulation-page">
            <div className="glass-panel benchmark-header-banner" style={{ marginBottom: "20px" }}>
              <h2 className="banner-title">Digital Twin Discrete-Time Simulation</h2>
              <p className="banner-subtitle">
                Interactive step-by-step playback with stochastic supplier lead-times and active disruption injection.
              </p>
            </div>

            <SimulationControls
              policy={policy}
              setPolicy={setPolicy}
              currentStep={currentStep}
              setCurrentStep={setCurrentStep}
              maxSteps={simData?.step_history_sample?.length || 20}
              isPlaying={isPlaying}
              setIsPlaying={setIsPlaying}
              speed={speed}
              setSpeed={setSpeed}
              onReset={handleReset}
              onRunSimulation={handleRunSimulation}
              loading={simLoading}
            />

            {simData && (
              <>
                <KpiMetricsGrid
                  kpis={simData.kpis}
                  costBreakdown={simData.cost_breakdown}
                />

                <NetworkVisualizer
                  currentStep={currentStep}
                  activeDisruptions={[]}
                  onHandUnits={currentStepData.on_hand_units || 2800}
                />

                <InventoryChart
                  stepHistory={simData.step_history_sample}
                  currentStep={currentStep}
                />
              </>
            )}

            {!simData && !simLoading && (
              <div className="glass-panel state-card" style={{ marginTop: "20px" }}>
                <p>Click "Run Episode" above to execute a fresh digital twin simulation run.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === "agent" && (
          <AgentChat
            kpis={simData?.kpis}
            onTriggerAction={(act) => {
              if (act.includes("MILP")) setActiveTab("optimization");
              else setActiveTab("overview");
            }}
          />
        )}
      </main>
    </div>
  );
}
