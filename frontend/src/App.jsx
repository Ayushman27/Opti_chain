import React, { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import KpiMetricsGrid from "./components/KpiMetricsGrid";
import SimulationControls from "./components/SimulationControls";
import NetworkVisualizer from "./components/NetworkVisualizer";
import InventoryChart from "./components/InventoryChart";
import BenchmarkView from "./components/BenchmarkView";
import ForecastView from "./components/ForecastView";
import AgentChat from "./components/AgentChat";
import { runSimulation, fetchBenchmark, fetchForecast, fetchHealth } from "./services/api";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("simulation");
  const [isOnline, setIsOnline] = useState(false);
  const [loading, setLoading] = useState(false);

  // Simulation state
  const [policy, setPolicy] = useState("min_max");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Simulation payload
  const [simData, setSimData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  const timerRef = useRef(null);

  // Initial load
  useEffect(() => {
    async function init() {
      const health = await fetchHealth();
      setIsOnline(health.status === "healthy");

      const simRes = await runSimulation("min_max", 90);
      setSimData(simRes);

      const benchRes = await fetchBenchmark();
      setBenchmarkData(benchRes);

      const fcRes = await fetchForecast();
      setForecastData(fcRes);
    }
    init();
  }, []);

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      const intervalMs = Math.max(100, Math.round(500 / speed));
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          const max = (simData?.step_history?.length || 90) - 1;
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

  const handleRunSimulation = async (selectedPolicy) => {
    setLoading(true);
    setIsPlaying(false);
    setCurrentStep(0);
    const res = await runSimulation(selectedPolicy, 90);
    setSimData(res);
    setLoading(false);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const currentStepData = simData?.step_history?.[currentStep] || {};
  const activeDisruptions = (simData?.disruptions || []).filter(
    (d) => currentStep >= d.start_step && currentStep < d.start_step + d.duration_steps
  );

  return (
    <div className="app-wrapper">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentStep={currentStep}
        episodeDays={simData?.step_history?.length || 90}
        isOnline={isOnline}
      />

      <main className="main-content">
        {activeTab === "simulation" && (
          <>
            <KpiMetricsGrid
              kpis={simData?.kpis}
              costBreakdown={simData?.cost_breakdown}
            />

            <SimulationControls
              policy={policy}
              setPolicy={setPolicy}
              currentStep={currentStep}
              setCurrentStep={setCurrentStep}
              maxSteps={simData?.step_history?.length || 90}
              isPlaying={isPlaying}
              setIsPlaying={setIsPlaying}
              speed={speed}
              setSpeed={setSpeed}
              onReset={handleReset}
              onRunSimulation={handleRunSimulation}
              loading={loading}
            />

            <NetworkVisualizer
              currentStep={currentStep}
              activeDisruptions={activeDisruptions}
              onHandUnits={currentStepData.on_hand_units || 2800}
            />

            <InventoryChart
              stepHistory={simData?.step_history}
              currentStep={currentStep}
            />
          </>
        )}

        {activeTab === "benchmark" && (
          <BenchmarkView benchmarkData={benchmarkData} />
        )}

        {activeTab === "forecasting" && (
          <ForecastView forecastData={forecastData} />
        )}

        {activeTab === "agent" && (
          <AgentChat
            kpis={simData?.kpis}
            onTriggerAction={(act) => {
              setActiveTab("simulation");
              if (act.includes("MILP")) setPolicy("milp");
            }}
          />
        )}
      </main>
    </div>
  );
}
