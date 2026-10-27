# OptiChain AI - Phase 1: Data Assumptions & Synthetic Generation Specification

This document details the mathematical assumptions, economic parameters, stochastic distributions, and execution instructions for **Phase 1: Business Problem & Synthetic Data Foundation**.

---

## 1. Business Problem & Operational Setting

The simulated enterprise is a multi-echelon retail distribution network tasked with fulfilling customer demand while minimizing cumulative supply chain costs. 

### Initial Topology (Phase 1 Baseline):
- **Warehouses**: 1 central regional distribution center (`WH_01`).
- **Product SKUs**: 5 distinct products spanning distinct categories (Electronics, FMCG, Apparel, Home Goods, Industrial).
- **Suppliers**: 3 primary tier-1 manufacturing/wholesale suppliers (`SUP_01` to `SUP_03`).
- **Simulated Horizon**: 90 discrete calendar days ($T=90$).

---

## 2. Mathematical Modeling & Data Generation Assumptions

### 2.1 Non-Stationary Customer Demand Model
Customer demand $D_{w, p, t} \in \mathbb{Z}_{\ge 0}$ for warehouse $w$ and product SKU $p$ at simulation day $t \in [0, T-1]$ is generated as an integer Poisson process whose rate parameter $\lambda(w, p, t)$ is governed by three non-stationary dynamics:

$$\lambda(w, p, t) = \text{BaseRate}_p \cdot \omega_w \cdot \text{Seasonality}(t) \cdot \text{Trend}(t) \cdot \prod_{d \in \mathcal{A}(t)} \text{Multiplier}_d$$
$$D_{w, p, t} \sim \text{Poisson}(\lambda(w, p, t))$$

Where:
1. **Base Demand Rate ($\text{BaseRate}_p$)**: Calibrated by product category:
   - **Fast-Moving Consumer Goods (FMCG)**: High volume, low variance ($\lambda \approx 25 - 60$ units/day).
   - **Electronics**: Low volume, high value ($\lambda \approx 4 - 15$ units/day).
   - **Apparel & General**: Medium velocity ($\lambda \approx 10 - 25$ units/day).
2. **Weekly Cyclical Seasonality ($\text{Seasonality}(t)$)**:
   $$\text{Seasonality}(t) = 1.0 + A_{\text{seas}} \cdot \cos\left(2\pi \cdot \frac{(t \pmod 7) - 5.5}{7}\right) + \beta_{\text{weekend}} \cdot \mathbb{I}((t \pmod 7) \in \{5, 6\})$$
   - Trough occurs mid-week (Wednesday, $dow=2$).
   - Peak occurs on weekend shopping cycles (Saturday & Sunday, $dow=5, 6$), boosted by $+20\%$.
3. **Long-Term Trend ($\text{Trend}(t)$)**:
   $$\text{Trend}(t) = 1.0 + \gamma_{\text{trend}} \cdot \left(\frac{t}{T}\right)$$
   Reflects quarterly baseline business growth ($\gamma_{\text{trend}} = 0.12$, i.e. $+12\%$ over 90 days).
4. **Stochastic Disruption Multiplier**:
   When a `DEMAND_SPIKE` disruption event is active at time $t$ targeting product $p$, $\lambda$ is multiplied by an impact factor (e.g. $2.5\times$ flash-sale surge).

---

### 2.2 Stochastic Supplier Lead Time Model
The replenishment delay $L_{s, t}$ (in days) between order issuance at day $t$ and shipment delivery is a truncated random variable:

$$L_{\text{raw}} \sim \mathcal{N}\left(\mu_L(s), \sigma_L(s)^2\right) + \Delta_{\text{anomaly}} \cdot \mathbb{I}(\text{Uniform}(0, 1) > \rho_s)$$
$$L_{s, t} = \max\left(1, \left\lfloor L_{\text{raw}} \cdot \prod_{d \in \mathcal{A}_s(t)} M_d + 0.5 \right\rfloor\right)$$

Where:
- $\mu_L(s) \in [2, 6]$ days: Base expected lead time.
- $\sigma_L(s) \in [0.4, 1.4]$ days: Operational freight variance.
- $\rho_s \in [0.88, 0.99]$: Delivery reliability probability. When a supplier experiences a reliability breach, an anomaly penalty $\Delta_{\text{anomaly}} \sim \text{Uniform}(1, 3)$ days is added.
- $M_d$: Disruption multiplier when a `SUPPLIER_DELAY` ($2.2\times$) or `PORT_CONGESTION` ($2.0\times$) event is active.
- **Physical Invariant**: Delivery lead time is strictly lower-bounded by $L \ge 1$ day.

---

### 2.3 Economic & Cost Structure

| Cost Element | Notation | Modeling Assumption | Typical Value Range |
|---|---|---|---|
| **Procurement Unit Cost** | $c_{p, s}$ | Supplier purchase price per unit | $\$2.00 - \$180.00$ / unit |
| **Selling Price** | $P_p$ | Commercial retail price | $c_p \times (1 + \text{margin})$ ($20\% - 70\%$ markup) |
| **Daily Holding Cost** | $h_p$ | Daily warehouse carrying charge (cost of capital, obsolescence, insurance) | $0.2\% - 0.5\%$ of unit cost / day |
| **Stockout Penalty** | $p_p$ | Opportunity loss + brand goodwill penalty per unmet customer unit | $1.2\times - 1.5\times$ unit cost |
| **Fixed Order Setup Cost** | $K_s$ | Fixed administrative & batching fee charged per supplier purchase order | $\$60.00 - \$200.00$ / PO |
| **Transportation Cost** | $\tau$ | Freight shipping charge per unit delivered | $\$1.50$ / unit |
| **Warehouse Operating Cost** | $F_w$ | Fixed daily facility maintenance & labor cost | $\$300.00 - \$750.00$ / day |

---

## 3. Disruption Scenarios (Exogenous Shocks)

1. **Viral Demand Spike (`DISR_01_DEMAND_SURGE`)**:
   - **Window**: Day 25 to Day 32 (duration: 7 days).
   - **Target**: Product SKU `PROD_001`.
   - **Impact**: Demand rate increases by $2.5\times$ ($+250\%$).
2. **Supplier Component Shortage (`DISR_02_SUPPLIER_DELAY`)**:
   - **Window**: Day 45 to Day 55 (duration: 10 days).
   - **Target**: Key Supplier `SUP_01`.
   - **Impact**: Lead time multiplied by $2.2\times$ (effective lead time spikes from 3 days to 7-8 days).

---

## 4. Software Design & Module Separation

The implementation strictly isolates configuration, entities, data generation, and tests into decoupled modules:

- **`optichain/config.py`**:
  Contains typed `SimulationConfig` dataclass with boundary validation (`n_warehouses >= 1`, `n_products >= 1`, `n_suppliers >= 1`, `episode_days >= 1`, `transport_cost_per_unit >= 0.0`).
- **`optichain/core/entities.py`**:
  Contains pure domain dataclasses: `Product`, `Warehouse`, `Supplier`, `Order`, `DisruptionEvent`. Every dataclass enforces domain rules inside `__post_init__` and implements `.to_dict()` serialization.
- **`optichain/data/generator.py`**:
  Implements `SupplyChainDataGenerator` containing standalone generators for products, warehouses, suppliers, disruptions, lead-time sampling, and 3D demand tensors.
- **`tests/test_data_foundation.py`**:
  Comprehensive test suite validating reproducibility, mathematical demand bounds, lead-time guarantees, and boundary validation.

---

## 5. How to Run the Tests

### 1. Run Phase 1 Foundation Tests Specifically:
```powershell
pytest tests/test_data_foundation.py -v
```

### 2. Run All Entity & Foundation Tests:
```powershell
pytest tests/test_data_foundation.py tests/test_entities.py -v
```

### 3. Run with Coverage Report:
```powershell
pytest --cov=optichain.data --cov=optichain.config --cov=optichain.core.entities tests/test_data_foundation.py -v
```

### 4. Run Entire Repository Test Suite:
```powershell
pytest tests -v
```
All 51 tests pass deterministically.
