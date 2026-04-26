# EcoGrid AI: Predictive, Prescriptive, and Generative Grid Dispatch ⚡🌿

**HackAZ 2026 Submission**

EcoGrid AI is an intelligent optimization pipeline designed to find the most effective way to lower CO₂ emissions in Arizona while strictly maintaining power grid stability. 

By taking in real-time environmental factors (Temperature and Solar Irradiance) and a target CO₂ cap, our system calculates the exact power generation mix required to meet regional demand, maximizing renewables safely before summarizing the strategy into an actionable report for grid operators.

## 🧠 The 3-Stage AI Architecture

Our project uniquely chains three different branches of Artificial Intelligence to solve the complex problem of energy dispatch:

1. **Predictive AI (Forecasting):** * **Model:** Multi-Layer Perceptron (MLPRegressor) via `scikit-learn`.
   * **Function:** Analyzes weather data (Temperature, GHI) to predict the hourly power demand (MW), as well as the maximum available Solar and Wind generation capacity.
2. **Prescriptive AI (Optimization):** * **Model:** Linear Programming via `PuLP`.
   * **Function:** Takes the predicted demand and renewable caps, and calculates the exact megawatt dispatch across 5 sources (Coal, Gas, Solar, Wind, Hydro). It mathematically guarantees the cheapest configuration that *meets demand* while *staying strictly under the user-defined CO₂ limit*.
3. **Generative AI (Reporting):** * **Model:** Meta LLaMA 3 (8B Instruct) via Hugging Face `transformers`.
   * **Function:** Grid operators aren't always data scientists. The LLM ingests the numerical optimization results and generates a plain-language, professional "Daily Energy Dispatch and Carbon Reduction Report" with actionable advisories.

## 📊 Data Sources

The model was trained on a custom dataset combining energy and environmental metrics for the Tucson, Arizona area:
* **Energy Information Administration (EIA):** Hourly grid demand and power generation data.
* **Environmental Protection Agency (EPA eGRID):** Emission factors for Coal and Natural Gas facilities.
* **National Renewable Energy Laboratory (NREL):** Solar intensity (GHI) and weather metrics.

**Key Dataset Features:**
* `Demand_MW`, `Solar_MW`, `Wind_MW`, `Hydro_MW`, `Coal_MW`, `Gas_MW`
* `GHI` (Global Horizontal Irradiance) & `Temperature`
* `Coal_CO2_lbs`, `Gas_CO2_lbs`, `Total_CO2`

## 💻 Tech Stack
* **Frontend:** HTML5, CSS3, JavaScript, Chart.js (Interactive UI with particle effects)
* **Backend:** Python, Flask, Flask-CORS
* **Machine Learning:** Scikit-Learn (MLP), Pandas, NumPy
* **Operations Research:** PuLP (Linear Optimization)
* **Generative AI:** PyTorch, Hugging Face Transformers (`meta-llama/Meta-Llama-3-8B-Instruct`)

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed, alongside a GPU-enabled environment if you intend to run the LLaMA 3 model locally.

### 2. Install Dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn pulp torch transformers