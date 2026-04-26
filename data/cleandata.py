import pandas as pd

def build_final_dataset():
    print("🚀 INITIALIZING REAL DATA PIPELINE...")

    # --- 1. LOAD YOUR EXISTING BALANCE FILE ---
    print("🔋 Loading EIA Grid Balance Data...")
    try:
        # Using the file you already have!
        df_grid = pd.read_csv('EIA930_BALANCE_2026_Jan_Jun.csv', low_memory=False)
    except FileNotFoundError:
        print("❌ ERROR: Could not find 'EIA930_BALANCE_2026_Jan_Jun.csv'.")
        return

    # Find the critical columns dynamically
    ba_col = [c for c in df_grid.columns if 'Authority' in c or 'Respondent' in c][0]
    time_col = [c for c in df_grid.columns if 'Time' in c and 'End' in c][0]
    
    # Filter immediately to Arizona specific grids to save memory (AZPS = Arizona Public Service, TEPC = Tucson Electric)
    df_grid = df_grid[df_grid[ba_col].str.contains('AZPS|TEPC', na=False)].copy()

    # Find the hidden fuel columns inside the Balance file
    coal_col = [c for c in df_grid.columns if 'from Coal' in c][0]
    gas_col = [c for c in df_grid.columns if 'from Natural Gas' in c][0]
    solar_gen_col = [c for c in df_grid.columns if 'from Solar' in c][0]
    nuclear_col = [c for c in df_grid.columns if 'from Nuclear' in c][0]
    wind_col = [c for c in df_grid.columns if 'from Wind' in c][0]
    hydro_col = [c for c in df_grid.columns if 'from Hydro' in c][0]
    demand_col = [c for c in df_grid.columns if 'Demand' in c and 'MW' in c][0]

    # Convert to numeric, replacing missing hours with 0
    df_grid['Coal_MW'] = pd.to_numeric(df_grid[coal_col], errors='coerce').fillna(0)
    df_grid['Gas_MW'] = pd.to_numeric(df_grid[gas_col], errors='coerce').fillna(0)
    df_grid['Solar_MW'] = pd.to_numeric(df_grid[solar_gen_col], errors='coerce').fillna(0)
    df_grid['Nuclear_MW'] = pd.to_numeric(df_grid[nuclear_col], errors='coerce').fillna(0)
    df_grid['Wind_MW'] = pd.to_numeric(df_grid[wind_col], errors='coerce').fillna(0)
    df_grid['Hydro_MW'] = pd.to_numeric(df_grid[hydro_col], errors='coerce').fillna(0)
    df_grid['Demand_MW'] = pd.to_numeric(df_grid[demand_col], errors='coerce').fillna(0)

    # --- 2. PRE-CALCULATE BASELINE EMISSIONS ---
    print("☁️ Calculating Current Carbon Footprint...")
    df_grid['Coal_CO2_lbs'] = df_grid['Coal_MW'] * 2200
    df_grid['Gas_CO2_lbs'] = df_grid['Gas_MW'] * 900
    df_grid['Total_CO2_lbs'] = df_grid['Coal_CO2_lbs'] + df_grid['Gas_CO2_lbs']

    # --- 3. FIX THE YEAR GAP (2026 vs 2023) ---
    df_grid['UTC Time'] = pd.to_datetime(df_grid[time_col])
    df_grid['Month'] = df_grid['UTC Time'].dt.month
    df_grid['Day'] = df_grid['UTC Time'].dt.day
    df_grid['Hour'] = df_grid['UTC Time'].dt.hour

    # --- 4. MERGE TUCSON SOLAR WEATHER ---
    print("☀️ Syncing Tucson Weather Data...")
    try:
        df_solar = pd.read_csv('NSRDB_Solar.csv')
    except FileNotFoundError:
        print("❌ ERROR: Could not find 'NSRDB_Solar.csv'.")
        return
        
    df_solar['timestamp'] = pd.to_datetime(df_solar['timestamp'])
    df_solar['Month'] = df_solar['timestamp'].dt.month
    df_solar['Day'] = df_solar['timestamp'].dt.day
    df_solar['Hour'] = df_solar['timestamp'].dt.hour
    
    # Keep only what we need to merge
    df_solar_clean = df_solar[['Month', 'Day', 'Hour', 'GHI', 'Temperature']]
    
    # The Merge
    final_df = pd.merge(df_grid, df_solar_clean, on=['Month', 'Day', 'Hour'], how='left')

    # --- 5. CLEANUP AND EXPORT ---
    print("🧹 Finalizing Export...")
    features = [
        'UTC Time', ba_col, 'Temperature', 'GHI', 'Demand_MW',
        'Nuclear_MW', 'Coal_MW', 'Gas_MW', 'Solar_MW', 'Wind_MW', 'Hydro_MW',
        'Coal_CO2_lbs', 'Gas_CO2_lbs', 'Total_CO2_lbs'
    ]
    
    ml_dataset = final_df[features].dropna()
    ml_dataset = ml_dataset.rename(columns={ba_col: 'Balancing_Authority'})
    
    print(f"\n✅ SUCCESS: {len(ml_dataset)} rows ready.")
    print("--- PREVIEW ---")
    print(ml_dataset.head(3))
    
    ml_dataset.to_csv('FINAL_ML_DATASET.csv', index=False)
    print("\n💾 Saved as 'FINAL_ML_DATASET.csv'. Ready to hand off to your partner!")

if __name__ == "__main__":
    build_final_dataset()