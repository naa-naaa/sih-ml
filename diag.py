# Diagnostics and exploratory data analysis script inspecting raw groundwater level, rainfall, and evapotranspiration records.
import pandas as pd

# Load your 3 CSVs
gwl = pd.read_csv(r'C:\Users\Admin\Downloads\groundwater_2018_2024 (1).csv')
rainfall = pd.read_csv(r'C:\Users\Admin\Downloads\rainfall_2018_2024.csv')
et = pd.read_csv(r'C:\Users\Admin\Downloads\evapo_2018_2024.csv')

# Filter to Ariyalur
gwl_ary = gwl[gwl['district'] == 'Ariyalur']
rain_ary = rainfall[rainfall['district'] == 'Ariyalur']
et_ary = et[et['districtName'] == 'Ariyalur']

print(f"GWL Ariyalur: {len(gwl_ary)} records")
print(f"Rainfall Ariyalur: {len(rain_ary)} records")
print(f"ET Ariyalur: {len(et_ary)} records")

# Convert to datetime
gwl_ary['dataTime'] = pd.to_datetime(gwl_ary['dataTime']).dt.date
rain_ary['dataTime'] = pd.to_datetime(rain_ary['dataTime']).dt.date
et_ary['date'] = pd.to_datetime(et_ary['date']).dt.date  # ET already has 'date'

# Unique days
print(f"\nUnique days - GWL: {len(gwl_ary['dataTime'].unique())}")
print(f"Unique days - Rainfall: {len(rain_ary['dataTime'].unique())}")
print(f"Unique days - ET: {len(et_ary['date'].unique())}")

# Check overlap among datasets
overlap = (
    set(gwl_ary['dataTime']) &
    set(rain_ary['dataTime']) &
    set(et_ary['date'])
)

print(f"\nDays in ALL 3 datasets: {len(overlap)}")

