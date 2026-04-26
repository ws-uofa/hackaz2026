
---

# ⚡ EcoGrid AI: Intelligent Grid Dispatch & Carbon Reduction

**A HackAZ 2026 Submission** *Optimizing Arizona's energy future through a trifecta of Predictive, Prescriptive, and Generative AI.*

---

## 🎯 The Mission
Arizona faces a unique energy challenge: intense solar potential paired with extreme temperature-driven demand. **EcoGrid AI** is an end-to-end optimization pipeline that ensures grid stability while aggressively pursuing CO₂ reduction. 

By analyzing real-time environmental factors (Temperature and Solar Irradiance), the system calculates the most sustainable power generation mix, stays strictly under a user-defined carbon cap, and translates complex optimization data into human-readable executive briefings.


## 🧠 The 3-Stage AI Architecture

EcoGrid AI doesn't just predict; it decides and communicates. Our architecture chains three distinct branches of AI:

1.  **Predictive AI (Forecasting):** * **Model:** Multi-Layer Perceptron (MLP) via `scikit-learn`.
    * **Function:** Ingests $Temp$ and $GHI$ (Global Horizontal Irradiance) to forecast hourly demand (MW) and the maximum potential of renewable sources.
2.  **Prescriptive AI (Optimization):** * **Model:** Linear Programming via `PuLP`.
    * **Function:** Mathematically guarantees the lowest-cost dispatch across Coal, Gas, Solar, Wind, and Hydro that meets demand while staying below the **CO₂ limit**.
3.  **Generative AI (Reporting):** * **Model:** Meta LLaMA 3 (8B Instruct) via Hugging Face.
    * **Function:** Ingests raw numerical outputs and generates a "Daily Energy Dispatch Report"—providing actionable advice for grid operators in plain language.

---

## 📂 Repository Structure

```text
.
├── app.py                # Flask Backend: Houses the MLP models, PuLP logic, and LLM pipeline
├── cleandata.py          # Data Engineering: Merges EIA grid data with NSRDB weather metrics
├── index.html            # Frontend: Interactive UI with Chart.js and particle animations
├── data/
│   ├── NSRDB_Solar.csv   # Raw solar & weather data (Tucson, AZ)
│   ├── EIA930_BALANCE... # Raw grid demand and fuel mix data
│   └── FINAL_ML_DATASET.csv # The cleaned, feature-engineered training set
└── README.md             # Project documentation
```

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript, **Chart.js**, Marked.js.
* **Backend:** Python, Flask, Flask-CORS.
* **Machine Learning:** Scikit-Learn (MLPRegressor), Pandas, NumPy.
* **Operations Research:** PuLP (CBC Solver).
* **Generative AI:** PyTorch, Hugging Face Transformers (**LLaMA 3 8B Instruct**).

---

## 🚀 Getting Started

### 1. Environment Setup
We recommend using a GPU-enabled environment (CUDA) for local LLM inference.
```bash
# Clone the repo
git clone https://github.com/your-repo/ecogrid-ai.git
cd ecogrid-ai

# Install dependencies
pip install flask flask-cors pandas numpy scikit-learn pulp torch transformers
```

### 2. Data Preparation
Run the cleaning script to sync weather patterns with grid demand:
```bash
python cleandata.py
```

### 3. Launch the Engine
Start the Flask server (this will load the LLaMA model into VRAM):
```bash
python app.py
```
*The server runs on `http://127.0.0.1:5005` by default.*

### 4. Open the Dashboard
Simply open `index.html` in any modern browser. Enter your weather parameters and CO₂ target to see the AI in action.

---

## 📊 Data Sources
Our models are grounded in real-world data specifically for the **Tucson, Arizona** region:
* **EIA (Energy Information Administration):** Hourly grid balance and fuel generation.
* **EPA (eGRID):** Carbon intensity factors for fossil fuel plants.
* **NREL (National Renewable Energy Laboratory):** Solar irradiance and local meteorological data.

---

## 👥 Authors
* **[Wilson Sun](https://wilsun.io)** - *ML & Backend Architecture*
* **[Kevin Burns](https://github.com/Kevin-Burns-UOFA)** - *Data Engineering & Optimization*
* **[Albert Tung](https://github.com/Albert-Tung)** - *Frontend & UI/UX Design*

---