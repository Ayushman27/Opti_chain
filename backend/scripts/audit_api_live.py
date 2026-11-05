"""
Script to exercise all FastAPI endpoints on the live running server
and record HTTP status codes, latencies, and payload properties.
"""
import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    t0 = time.time()
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "AuditScript/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            return resp.status, json.loads(data), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        data = e.read().decode("utf-8")
        return e.code, data, elapsed

def post(path, body):
    t0 = time.time()
    url = f"{BASE_URL}{path}"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "AuditScript/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            return resp.status, json.loads(data), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        data = e.read().decode("utf-8")
        return e.code, data, elapsed

def main():
    print(f"Connecting to {BASE_URL}...")
    
    # 1. Health
    status, data, el = get("/api/v1/health")
    print(f"1. GET /api/v1/health -> {status} in {el:.3f}s: {data}")
    assert status == 200, f"Expected 200, got {status}"
    
    # 2. Dashboard summary
    status, data, el = get("/api/v1/dashboard/summary")
    print(f"2. GET /api/v1/dashboard/summary -> {status} in {el:.3f}s: Cost=${data.get('total_cost')}, Profit=${data.get('net_profit')}")
    assert status == 200
    
    # 3. Benchmarks
    status, data, el = get("/api/v1/benchmarks")
    print(f"3. GET /api/v1/benchmarks -> {status} in {el:.3f}s: {len(data.get('policies', []))} policies")
    assert status == 200
    
    # 4. Forecasts
    status, data, el = get("/api/v1/forecasts?product_id=PROD_001&model=Ridge&horizon=14")
    print(f"4. GET /api/v1/forecasts -> {status} in {el:.3f}s: model={data.get('model_name')}, metrics={data.get('metrics')}")
    assert status == 200

    # 5. Inventory
    status, data, el = get("/api/v1/inventory")
    print(f"5. GET /api/v1/inventory -> {status} in {el:.3f}s: {len(data.get('products', []))} SKUs, on_hand={data.get('total_on_hand_units')}")
    assert status == 200

    # 6. Optimization recommendations
    opt_body = {
        "horizon": 7,
        "safety_buffer_factor": 0.2,
        "time_limit_sec": 5,
        "forecast_method": "ridge"
    }
    status, data, el = post("/api/v1/optimization/recommendations", opt_body)
    print(f"6. POST /api/v1/optimization/recommendations -> {status} in {el:.3f}s: solver_status={data.get('solver_status')}, proven_optimal={data.get('is_proven_optimal')}, orders={len(data.get('recommended_orders', []))}")
    assert status == 200

    # 7. Simulation run
    sim_body = {
        "policy": "min_max",
        "episode_days": 15,
        "random_seed": 42
    }
    status, data, el = post("/api/v1/simulation/run", sim_body)
    print(f"7. POST /api/v1/simulation/run -> {status} in {el:.3f}s: run_id={data.get('run_id')}, fill_rate={data.get('kpis', {}).get('fill_rate_percent')}%")
    assert status == 200

    print("ALL 7 ENDPOINTS RETURNED HTTP 200 OK!")

if __name__ == "__main__":
    main()
