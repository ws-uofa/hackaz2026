import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pulp
import warnings

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Predictive AI: Train the Model (Offline Phase)
# ==========================================
class PredictiveSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        # Using MLP (Multi-Layer Perceptron) to predict Demand and Renewable potential
        self.model_demand = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.model_solar = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        self.model_wind = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        
    def train(self, df):
        print(">>> Training Predictive AI (MLP) on historical CSV data...")
        X = df[['Temperature', 'GHI']]
        X_scaled = self.scaler.fit_transform(X)
        
        # We treat historical actuals as the "maximum potential" for training purposes
        self.model_demand.fit(X_scaled, df['Demand_MW'])
        self.model_solar.fit(X_scaled, df['Solar_MW'])
        self.model_wind.fit(X_scaled, df['Wind_MW'])
        print(">>> Training Complete!")
        
    def predict(self, temperature, ghi):
        X_input = self.scaler.transform([[temperature, ghi]])
        pred_demand = self.model_demand.predict(X_input)[0]
        pred_solar_max = self.model_solar.predict(X_input)[0]
        pred_wind_max = self.model_wind.predict(X_input)[0]
        
        # Ensure predictions are non-negative
        return max(0, pred_demand), max(0, pred_solar_max), max(0, pred_wind_max)

# ==========================================
# 2. Prescriptive AI: Operations Research Solver (Online Phase)
# ==========================================
def prescriptive_optimizer(pred_demand, pred_solar_max, pred_wind_max, target_co2, ef_coal, ef_gas, max_hydro):
    print(">>> Running Prescriptive AI (PuLP Solver) for optimal dispatch...")
    
    # Assume maximum installed capacity for traditional energy sources
    max_coal = 5000 
    max_gas = 4000
    
    prob = pulp.LpProblem("Carbon_Constrained_Dispatch", pulp.LpMinimize)
    
    # Decision Variables (Lower bound is 0, Upper bound is max capacity or predicted limit)
    p_coal = pulp.LpVariable("Coal_MW", 0, max_coal)
    p_gas = pulp.LpVariable("Gas_MW", 0, max_gas)
    p_solar = pulp.LpVariable("Solar_MW", 0, pred_solar_max) 
    p_wind = pulp.LpVariable("Wind_MW", 0, pred_wind_max)   
    p_hydro = pulp.LpVariable("Hydro_MW", 0, max_hydro)
    
    # Objective Function: Minimize cost (High cost for fossil fuels, low/zero for renewables)
    prob += 50 * p_coal + 60 * p_gas + 5 * p_solar + 5 * p_wind + 5 * p_hydro
    
    # Constraint 1: Supply must meet or exceed predicted demand
    prob += (p_coal + p_gas + p_solar + p_wind + p_hydro) >= pred_demand, "Demand_Constraint"
    
    # Constraint 2: Total carbon emissions must be less than or equal to the target
    prob += (p_coal * ef_coal + p_gas * ef_gas) <= target_co2, "Carbon_Constraint"
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != 1:
        return None, "Optimization Failed: The carbon target is too low or demand is too high to solve physically."
        
    results = {
        'Coal_MW': p_coal.varValue, 
        'Gas_MW': p_gas.varValue,
        'Solar_MW': p_solar.varValue, 
        'Wind_MW': p_wind.varValue,
        'Hydro_MW': p_hydro.varValue,
        'Total_CO2': p_coal.varValue * ef_coal + p_gas.varValue * ef_gas
    }
    return results, "Success"

# ==========================================
# 3. Generative AI: Constructing the Dynamic Prompt
# ==========================================
def generative_ai_prompt(temperature, ghi, target_co2, predictions, opt_results):
    print(">>> Generating the Prompt for the LLM...")
    
    prompt = f"""
You are an expert in power grid dispatch and carbon emission optimization. Based on the system-generated pipeline data below, write a professional "Daily Energy Dispatch and Carbon Reduction Report".

[1. Boundary Conditions Forecasted by Predictive AI]
- Weather Forecast: Temperature {temperature}°C, GHI {ghi}
- Predicted Total Grid Demand: {predictions['Demand']:.1f} MW
- Predicted Solar Available Potential: {predictions['Solar_Max']:.1f} MW
- Predicted Wind Available Potential: {predictions['Wind_Max']:.1f} MW

[2. User-Defined Carbon Reduction Target]
- Maximum Carbon Emission Target: {target_co2:,.0f} lbs

[3. Optimal Strategy Calculated by Prescriptive AI]
To meet the demand and carbon targets, the operations research engine has mandated the following dispatch combination:
- Solar Output: {opt_results['Solar_MW']:.1f} MW
- Wind Output: {opt_results['Wind_MW']:.1f} MW
- Hydro Output: {opt_results['Hydro_MW']:.1f} MW
- Natural Gas Output: {opt_results['Gas_MW']:.1f} MW
- Coal Output: {opt_results['Coal_MW']:.1f} MW
- Expected Actual Total CO2 Emissions: {opt_results['Total_CO2']:,.0f} lbs

[Your Task]
Please generate a structured report containing the following sections:
1. **Forecast Briefing**: Explain the relationship between the forecasted weather, the grid demand, and renewable energy potential.
2. **Optimal Dispatch Strategy Explanation**: Explain *why* the algorithm made this specific allocation (e.g., maximizing solar due to high GHI, using gas to bridge the gap while suppressing coal to meet the carbon target).
3. **Grid Operations Advisory**: Provide 2 actionable recommendations for grid operators to mitigate stability risks associated with renewable energy fluctuations today.

Do not alter the baseline numbers provided above. Base your analysis strictly on the "Optimal Strategy Calculated by Prescriptive AI".
"""
    return prompt.strip()

# ==========================================
# 4. Master Pipeline
# ==========================================
def run_pipeline(csv_path, target_co2, current_temp, current_ghi):
    # --- Data Loading & Preprocessing ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_path}'. Please ensure the path is correct.")
        return
    
    # Calculate average emission factors (lbs/MW) dynamically from historical data
    # Ignoring days where MW was 0 to avoid division by zero
    valid_coal = df[df['Coal_MW'] > 0]
    ef_coal = (valid_coal['Coal_CO2_lbs'] / valid_coal['Coal_MW']).mean()
    
    valid_gas = df[df['Gas_MW'] > 0]
    ef_gas = (valid_gas['Gas_CO2_lbs'] / valid_gas['Gas_MW']).mean()
    
    # Determine hydro capacity based on historical max
    max_hydro = df['Hydro_MW'].max()

    # --- Offline Phase (Training) ---
    predictor = PredictiveSystem()
    predictor.train(df)
    
    # --- Online Phase (System Input) ---
    print(f"\n[System Input] Temp: {current_temp}°C, GHI: {current_ghi}, Target CO2: {target_co2:,.0f} lbs")
    
    # Step A: Predict
    pred_demand, pred_solar, pred_wind = predictor.predict(current_temp, current_ghi)
    predictions = {'Demand': pred_demand, 'Solar_Max': pred_solar, 'Wind_Max': pred_wind}
    
    # Step B: Optimize
    opt_results, status = prescriptive_optimizer(
        pred_demand, pred_solar, pred_wind, target_co2, ef_coal, ef_gas, max_hydro
    )
    
    if opt_results is None:
        print(status)
        return
        
    # Step C: Generate
    final_prompt = generative_ai_prompt(current_temp, current_ghi, target_co2, predictions, opt_results)
    
    print("\n================ [Final Prompt (Ready for LLM)] ================\n")
    print(final_prompt)

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Example usage: Replace 'data.csv' with your actual file path.
    # Set tomorrow's expected temperature, GHI, and your firm carbon cap.
    
    CSV_FILE_PATH = "data.csv"
    TARGET_CO2_LBS = 4500000 
    TOMORROW_TEMP = 28.5
    TOMORROW_GHI = 820

    # run_pipeline(CSV_FILE_PATH, TARGET_CO2_LBS, TOMORROW_TEMP, TOMORROW_GHI)
    print("Code is ready. Uncomment the run_pipeline execution line to test with your actual CSV file.")