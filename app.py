from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pulp
import warnings
import torch
from transformers import pipeline

# 抑制一些烦人的警告信息
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) # 允许前端请求跨域

# ==========================================
# 1. 预测系统 (Predictive AI)
# ==========================================
class PredictiveSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model_demand = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.model_solar = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        self.model_wind = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        
    def train(self, df):
        print(">>> Training Predictive AI (MLP)...")
        X = df[['Temperature', 'GHI']]
        X_scaled = self.scaler.fit_transform(X)
        self.model_demand.fit(X_scaled, df['Demand_MW'])
        self.model_solar.fit(X_scaled, df['Solar_MW'])
        self.model_wind.fit(X_scaled, df['Wind_MW'])
        print(">>> Predictive AI Training Complete!")
        
    def predict(self, temperature, ghi):
        X_input = self.scaler.transform([[temperature, ghi]])
        pred_demand = self.model_demand.predict(X_input)[0]
        pred_solar_max = self.model_solar.predict(X_input)[0]
        pred_wind_max = self.model_wind.predict(X_input)[0]
        return max(0, pred_demand), max(0, pred_solar_max), max(0, pred_wind_max)

# ==========================================
# 2. 优化系统 (Prescriptive AI)
# ==========================================
def prescriptive_optimizer(pred_demand, pred_solar_max, pred_wind_max, target_co2, ef_coal, ef_gas, max_hydro):
    print(">>> Running Optimization (PuLP)...")
    max_coal, max_gas = 5000, 4000
    prob = pulp.LpProblem("Carbon_Constrained_Dispatch", pulp.LpMinimize)
    
    p_coal = pulp.LpVariable("Coal_MW", 0, max_coal)
    p_gas = pulp.LpVariable("Gas_MW", 0, max_gas)
    p_solar = pulp.LpVariable("Solar_MW", 0, pred_solar_max) 
    p_wind = pulp.LpVariable("Wind_MW", 0, pred_wind_max)   
    p_hydro = pulp.LpVariable("Hydro_MW", 0, max_hydro)
    
    prob += 50 * p_coal + 60 * p_gas + 5 * p_solar + 5 * p_wind + 5 * p_hydro
    prob += (p_coal + p_gas + p_solar + p_wind + p_hydro) >= pred_demand
    prob += (p_coal * ef_coal + p_gas * ef_gas) <= target_co2
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != 1:
        return None, "Optimization Failed: The carbon target is too low or demand is too high."
        
    return {
        'Coal_MW': p_coal.varValue, 'Gas_MW': p_gas.varValue,
        'Solar_MW': p_solar.varValue, 'Wind_MW': p_wind.varValue,
        'Hydro_MW': p_hydro.varValue,
        'Total_CO2': p_coal.varValue * ef_coal + p_gas.varValue * ef_gas
    }, "Success"

def generative_ai_prompt(temperature, ghi, target_co2, predictions, opt_results):
    # 构建给大模型的系统提示词
    return f"""
You are an expert in grid dispatch. Write a professional "Daily Energy Dispatch and Carbon Reduction Report".

[Forecast] Temp {temperature}°C, GHI {ghi}. Demand: {predictions['Demand']:.1f} MW.
[Target] CO2 Limit: {target_co2:,.0f} lbs.
[Dispatch] Solar: {opt_results['Solar_MW']:.1f} MW, Wind: {opt_results['Wind_MW']:.1f} MW, Hydro: {opt_results['Hydro_MW']:.1f} MW, Gas: {opt_results['Gas_MW']:.1f} MW, Coal: {opt_results['Coal_MW']:.1f} MW. Expected CO2: {opt_results['Total_CO2']:,.0f} lbs.

Generate a structured report:
1. Forecast Briefing: Explain weather impact.
2. Strategy Explanation: Why this allocation?
3. Advisory: 2 actionable tips for grid operators.
"""

# ==========================================
# 3. 全局初始化 (启动服务器时加载一次数据和模型)
# ==========================================
CSV_FILE_PATH = "./cleaned_daily_energy_data.csv" # 确保这个路径正确

print("\n" + "="*50)
print("1/2: Loading Data and Training Predictive AI...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    valid_coal = df[df['Coal_MW'] > 0]
    ef_coal = (valid_coal['Coal_CO2_lbs'] / valid_coal['Coal_MW']).mean()
    valid_gas = df[df['Gas_MW'] > 0]
    ef_gas = (valid_gas['Gas_CO2_lbs'] / valid_gas['Gas_MW']).mean()
    max_hydro = df['Hydro_MW'].max()

    predictor = PredictiveSystem()
    predictor.train(df)
except Exception as e:
    print(f"⚠️ Error loading CSV: {e}. Please check your CSV path.")

print("\n2/2: Loading Hugging Face LLM into GPU... (Takes a few minutes)")
try:
    # 你可以把这里换成 "meta-llama/Meta-Llama-3-8B-Instruct" 如果你想用 Llama
    # 如果你之前不是存在默认缓存，而是存在了特定的文件夹，把名字换成绝对路径即可
    MODEL_ID = "/data1/public/hf/meta-llama/Meta-Llama-3-8B-Instruct"
    
    llm_pipeline = pipeline(
        "text-generation", 
        model=MODEL_ID, 
        device_map="auto",             # 自动寻找 GPU
        torch_dtype=torch.float16      # 使用半精度节省显存
    )
    print(">>> All Systems GO! Server is Ready.")
except Exception as e:
    print(f"⚠️ LLM Loading Error: {e}")
    llm_pipeline = None
print("="*50 + "\n")


# ==========================================
# 4. Flask API 路由配置
# ==========================================
@app.route('/api/dispatch', methods=['POST'])
def api_dispatch():
    data = request.json
    print("successfully get!!!!!!!!!!!!")
    current_temp = float(data.get('temp', 28.5))
    current_ghi = float(data.get('ghi', 820))
    target_co2 = float(data.get('co2', 4500000))
    
    # 执行预测和优化
    pred_demand, pred_solar, pred_wind = predictor.predict(current_temp, current_ghi)
    predictions = {'Demand': pred_demand, 'Solar_Max': pred_solar, 'Wind_Max': pred_wind}
    opt_results, status = prescriptive_optimizer(
        pred_demand, pred_solar, pred_wind, target_co2, ef_coal, ef_gas, max_hydro
    )
    
    if opt_results is None:
        return jsonify({"error": status}), 400
        
    final_prompt = generative_ai_prompt(current_temp, current_ghi, target_co2, predictions, opt_results)
    
    if llm_pipeline is None:
        return jsonify({"error": "LLM failed to load on server."}), 500

    # 调用 Hugging Face 模型生成文本
    try:
        print("\n>>> Generating report locally with Hugging Face...")
        
        # 给指令模型套上系统对话模板（这样输出质量更好）
        messages = [
            {"role": "system", "content": "You are a professional energy data analyst."},
            {"role": "user", "content": final_prompt}
        ]
        
        # 让 pipeline 自动帮我们生成回答
        outputs = llm_pipeline(
            messages, 
            max_new_tokens=600,   # 最大生成长度
            temperature=0.7,      # 发散程度
            do_sample=True
        )
        
        # 提取模型生成的文本内容
        report_text = outputs[0]["generated_text"][-1]["content"]
        
        return jsonify({"report": report_text})
        
    except Exception as e:
        print(f"LLM Inference Error: {e}")
        return jsonify({"error": "Failed to generate report using HF model."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=False) # 在跑大模型时，建议把 debug 设为 False