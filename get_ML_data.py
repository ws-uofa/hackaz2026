import pandas as pd

# 1. Load the full dataset
# Replace with your actual file name
df = pd.read_csv('FINAL_ML_DATASET.csv')

# 2. Extract the Date for daily grouping
df['Date'] = pd.to_datetime(df['UTC Time']).dt.date

# 3. TWEAK: Fix the negative CO2 mathematical artifact
# Plants drawing power do not absorb CO2, so we set any negatives to 0 before summing
co2_columns = ['Coal_CO2_lbs', 'Gas_CO2_lbs', 'Total_CO2_lbs']
for col in co2_columns:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# 4. Drop unnecessary columns
# Keep Balancing_Authority! (errors='ignore' prevents crashes if Nuclear_MW is already gone)
df = df.drop(columns=['UTC Time', 'Nuclear_MW'], errors='ignore')

# 5. Set up the specific aggregation rules
agg_dict = {
    'Temperature': 'mean',
    'GHI': 'sum',
    'Demand_MW': 'sum',
}

# 6. TWEAK: Dynamically add remaining columns ONLY if they are numeric
for col in df.columns:
    if col not in agg_dict and col not in ['Date', 'Balancing_Authority']:
        # This prevents crashes if non-number columns exist
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = 'sum'

# 7. TWEAK: Group by BOTH Date and Balancing Authority
df_daily = df.groupby(['Date', 'Balancing_Authority']).agg(agg_dict).reset_index()

# 8. Save the cleaned data
output_filename = 'cleaned_daily_energy_data.csv'
df_daily.to_csv(output_filename, index=False)

print(f"Success! Cleaned data saved to {output_filename}.")
print(f"Total rows generated: {len(df_daily)}")