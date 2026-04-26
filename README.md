# hackaz2026

### Predictive AI -> Prescriptive AI -> Generative AI

Data sets gathered from:

https://www.eia.gov/
https://www.epa.gov/egrid
https://nsrdb.nlr.gov/


-----Goals for the Project-------------------------------------------------

The goal of this project is to build a Linear Regression ML model that finds the best way to lower CO2 emissions in Arizona while keeping our power output stable.

I wanted to build a tool where a user can input a CO2 target and a Power Demand, and the model will return the specific % of Renewable Energy that needs to be developed to hit those goals. The model is designed to be realistic—it finds the minimum emissions without completely killing off coal and gas, which the grid still needs for reliability.


----How it Works-----------------------------------------------------------

ML Model: We use Linear Regression to map the relationship between power demand, weather constraints (GHI/Temp), and carbon output.

Optimization: The model acts as a "Reverse Estimator"—you give it the CO2 limit you want, and it tells you what percentage of the grid needs to be renewable to make that happen.

--- User input --------------------------------------------------------------------

Idea user will input a current CO2 input and power demand. model will return the amount(% of renewable energy) That needs to be developed to minizize the emission and maximun the power. (should not completely get rid of coal and gas)


-- Visual ------------------------------------------------------------------

The project outputs bar charts that show the full breakdown of the final percentages. You can see exactly how the "Green Shift" looks compared to the current fossil fuel heavy mix.


--ABOUT DATA---------------------------------------------------------------

The dataset combines 2026 hourly grid data from the EIA and 2023 solar intensity data from NREL for the Tucson area.

Renewable data

solar_MW -- power output for solar
Hydro_MW -- power output for hydro
Wind_MW -- power output of wind
Nuclear_MW --- Null as there was no data points

Non Renewable

Coal_MW -- power output of coal
Gas_Mw -- power output of gas

Others 

Demand_MW -- The demand of power at the hour
GHI -- is the amount of sunshine on a solar pannel
Temperature -- the temp at the our
Balancing_Authority -- power district
UTC_Time -- time 
Coal_CO2 -- CO2 of coal production
Gas_CO2 -- CO2 of gas production
Total_CO2 -- total CO2
