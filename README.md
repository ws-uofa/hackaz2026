# hackaz2026

### Predictive AI -> Prescriptive AI -> Generative AI

Data sets gathered from:

https://www.eia.gov/
https://www.epa.gov/egrid
https://nsrdb.nlr.gov/

-----Goals for the Project-------------------------------------------------

The goal of this project is to build an AI-driven optimization pipeline that identifies the most effective way to lower CO2 emissions in Arizona while maintaining strict power grid stability.

This system provides predictive forecasting and prescriptive recommendations, allowing users to input a target CO2 emission reduction and receive an actionable, data-backed plan. The model calculates how to adjust power generation outputs—maximizing renewable utilization based on real-time environmental factors—without compromising the energy demands of the region.

----How it Works------------------------------------------------------------------

Predictive model, we train the model on a data set to predict the trends in Arizona and its power usage along with CO2 emission.

Prescriptive, we draw planning using MLP (multi-layer perceptron) from the training data. It outputs the ideal percentage of renewable versus non-renewable generation required to hit the target safely, ensuring baseline stability with gas and coal when necessary.

Generative AI, An LLM interprets the prescriptive model’s numeric outputs, generating a plain-language, actionable summary of the required grid changes and projected emission reductions for stakeholders.

--- User input --------------------------------------------------------------------
Idea user will input a current CO2 input and power demand. model will return the amount (% of renewable energy) That needs to be developed to minimize the emission and maximize the power. (should not completely get rid of coal and gas)

-- Visual ------------------------------------------------------------------
The output includes a visual breakdown of the projected emissions and the specific grid changes required.

--ABOUT DATA--
The dataset combines 2026 hourly grid data from the EIA and 2023 solar intensity data from NREL for the Tucson area.

Renewable data
solar_MW -- power output for solar
Hydro_MW -- power output for hydro
Wind_MW -- power output of wind

Non Renewable
Coal_MW -- power output of coal
Gas_MW -- power output of gas

Others
Demand_MW -- The demand of power at the hour
GHI -- is the amount of sunshine on a solar panel
Temperature -- the temp
Balancing_Authority -- power district
Date -- Day of the year
Coal_CO2 -- CO2 of coal production
Gas_CO2 -- CO2 of gas production
Total_CO2 -- total CO2
