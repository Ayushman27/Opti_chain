import React from "react";
import { Play, Pause, RotateCcw, FastForward, Sliders, RefreshCw, PlayCircle } from "lucide-react";

export default function SimulationControls({
  policy,
  setPolicy,
  currentStep,
  setCurrentStep,
  maxSteps,
  isPlaying,
  setIsPlaying,
  speed,
  setSpeed,
  onReset,
  onRunSimulation,
  loading,
  hasData = false,
}) {
  const policies = [
    { id: "min_max", name: "Min-Max (s, S) Policy" },
    { id: "rop", name: "Reorder Point (ROP/EOQ)" },
    { id: "periodic", name: "Periodic Review (R, S)" },
    { id: "forecast", name: "ML Forecast Dynamic Buffer" },
    { id: "milp", name: "Constrained MILP (PuLP MPC)" },
  ];

  const handlePlayToggle = () => {
    if (!hasData) {
      onRunSimulation(policy);
      setIsPlaying(true);
      return;
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="glass-panel sim-controls-card">
      <div className="controls-row-top">
        {/* Policy Selector & Run Button */}
        <div className="control-group">
          <label className="control-label">
            <Sliders size={14} /> Control Policy Engine
          </label>
          <div className="policy-select-wrapper">
            <select
              value={policy}
              onChange={(e) => {
                setPolicy(e.target.value);
                onRunSimulation(e.target.value);
              }}
              disabled={loading}
              className="policy-select"
            >
              {policies.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>

            <button
              onClick={() => onRunSimulation(policy)}
              className="btn-run-episode"
              disabled={loading}
              title="Execute fresh 90-day simulation episode"
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              <span>{loading ? "Simulating..." : "Run Episode"}</span>
            </button>
          </div>
        </div>

        {/* Playback Action Buttons */}
        <div className="playback-actions">
          <button
            onClick={handlePlayToggle}
            className={`btn-playback ${isPlaying ? "btn-pause" : "btn-play"}`}
            disabled={loading}
            title={hasData ? (isPlaying ? "Pause playback" : "Play step-by-step") : "Run and Play Simulation"}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            <span>{isPlaying ? "Pause" : "Play"}</span>
          </button>

          <button
            onClick={() => {
              if (currentStep < maxSteps - 1) setCurrentStep((prev) => prev + 1);
            }}
            className="btn-step"
            disabled={isPlaying || currentStep >= maxSteps - 1 || loading || !hasData}
            title="Advance 1 simulation day"
          >
            <FastForward size={16} />
            <span>Step +1</span>
          </button>

          <button
            onClick={onReset}
            className="btn-reset"
            disabled={loading}
            title="Reset to Day 0"
          >
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
        </div>

        {/* Speed Multiplier */}
        <div className="speed-toggle-group">
          {[0.5, 1, 2, 5].map((spd) => (
            <button
              key={spd}
              onClick={() => setSpeed(spd)}
              className={`btn-speed ${speed === spd ? "active" : ""}`}
            >
              {spd}x
            </button>
          ))}
        </div>
      </div>

      {/* Scrubbing Timeline Slider */}
      <div className="scrubber-container">
        <div className="scrubber-header">
          <span className="scrubber-title">Simulation Progress Timeline</span>
          <span className="scrubber-value">
            Day <strong>{currentStep}</strong> of {maxSteps}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(1, maxSteps - 1)}
          value={currentStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          disabled={!hasData}
          className="timeline-slider"
        />
        <div className="timeline-ticks">
          <span>Day 0 (Warm-up)</span>
          <span className="disruption-marker">Day 25 (Viral Surge)</span>
          <span className="disruption-marker">Day 45 (Supplier Delay)</span>
          <span className="disruption-marker">Day 68 (Port Congestion)</span>
          <span>Day {maxSteps}</span>
        </div>
      </div>
    </div>
  );
}
