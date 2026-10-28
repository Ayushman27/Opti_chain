/**
 * OptiChain AI - Frontend REST API Client
 * Centralized HTTP service communicating with the FastAPI backend.
 * Configured via Vite environment variable VITE_API_BASE_URL.
 * No fake mock data: honest error and loading propagation.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined 
  ? import.meta.env.VITE_API_BASE_URL 
  : "http://localhost:8000";

/**
 * Helper to handle HTTP errors with helpful message parsing.
 */
async function handleResponse(res, endpointDesc) {
  if (!res.ok) {
    let errorDetail = "";
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
    } catch {
      errorDetail = await res.text();
    }
    throw new Error(`${endpointDesc} failed (HTTP ${res.status}): ${errorDetail || res.statusText}`);
  }
  return await res.json();
}

/**
 * 1. Health check & dependency verification
 */
export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`);
  return await handleResponse(res, "Health check");
}

/**
 * 2. Executive Dashboard KPI summary
 */
export async function fetchDashboardSummary() {
  const res = await fetch(`${API_BASE_URL}/api/v1/dashboard/summary`);
  return await handleResponse(res, "Dashboard summary fetch");
}

/**
 * 3. Benchmark comparison matrix
 */
export async function fetchBenchmarks() {
  const res = await fetch(`${API_BASE_URL}/api/v1/benchmarks`);
  return await handleResponse(res, "Benchmarks fetch");
}

/**
 * 4. Demand forecasting actuals and projections
 */
export async function fetchForecasts({ productId, model, horizon = 14 } = {}) {
  const params = new URLSearchParams();
  if (productId) params.append("product_id", productId);
  if (model) params.append("model", model);
  if (horizon) params.append("horizon", horizon.toString());

  const url = `${API_BASE_URL}/api/v1/forecasts${params.toString() ? `?${params.toString()}` : ""}`;
  const res = await fetch(url);
  return await handleResponse(res, "Forecasts fetch");
}

/**
 * 5. Digital Twin product-level inventory status
 */
export async function fetchInventory() {
  const res = await fetch(`${API_BASE_URL}/api/v1/inventory`);
  return await handleResponse(res, "Inventory fetch");
}

/**
 * 6. MILP Replenishment Optimization Recommendations
 */
export async function fetchOptimizationRecommendations({
  horizon = 7,
  safetyBufferFactor = 0.2,
  timeLimitSec = 5,
  forecastMethod = "ridge",
  productIds = null,
} = {}) {
  const payload = {
    horizon: Number(horizon),
    safety_buffer_factor: Number(safetyBufferFactor),
    time_limit_sec: Number(timeLimitSec),
    forecast_method: forecastMethod,
  };
  if (productIds && productIds.length > 0) {
    payload.product_ids = productIds;
  }

  const res = await fetch(`${API_BASE_URL}/api/v1/optimization/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await handleResponse(res, "Optimization recommendations solve");
}

/**
 * 7. Controlled Simulation Run
 */
export async function executeSimulation({
  policy = "min_max",
  episodeDays = 90,
  randomSeed = 42,
  nWarehouses = 1,
  nProducts = 5,
  nSuppliers = 3,
} = {}) {
  const res = await fetch(`${API_BASE_URL}/api/v1/simulation/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      policy,
      episode_days: episodeDays,
      random_seed: randomSeed,
      n_warehouses: nWarehouses,
      n_products: nProducts,
      n_suppliers: nSuppliers,
    }),
  });
  return await handleResponse(res, "Simulation run");
}

/**
 * Reasoning Agent Chat
 */
export async function queryAgent(prompt, contextKpis = {}) {
  const res = await fetch(`${API_BASE_URL}/api/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      context_kpis: contextKpis,
    }),
  });
  return await handleResponse(res, "Agent chat query");
}

export { API_BASE_URL };
