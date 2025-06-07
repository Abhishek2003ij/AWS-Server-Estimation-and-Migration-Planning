import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv("/pods_with_servers_needed.csv") 


X = df[["cpu_milli", "memory_mib", "num_gpu", "gpu_milli"]]
y = df["servers_needed"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)


xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("Random Forest:")
print("  RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("  R^2:", r2_score(y_test, rf_preds))

print("\nXGBoost:")
print("  RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("  R^2:", r2_score(y_test, xgb_preds))


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_preds, color='blue', label='RF Prediction')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')
plt.legend()

# XGB Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, xgb_preds, color='green', label='XGB Prediction')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost')
plt.legend()

plt.suptitle("Actual vs Predicted Servers Needed")
plt.tight_layout()
plt.show()

AWS_INSTANCES = {
    't3.large':     {'cpu_milli': 2000,  'memory_mib': 8192,   'gpu_milli': 0},
    'm5.2xlarge':   {'cpu_milli': 8000,  'memory_mib': 32768,  'gpu_milli': 0},
    'c5.4xlarge':   {'cpu_milli': 16000, 'memory_mib': 32768,  'gpu_milli': 0},
    'g4dn.4xlarge': {'cpu_milli': 16000, 'memory_mib': 65536,  'gpu_milli': 1000},
    'g4dn.12xlarge':{'cpu_milli': 48000, 'memory_mib': 192000, 'gpu_milli': 4000},
}


def estimate_aws_servers(row, spec):
    cpu = row['cpu_milli'] / spec['cpu_milli'] if spec['cpu_milli'] else np.inf
    mem = row['memory_mib'] / spec['memory_mib'] if spec['memory_mib'] else np.inf
    if row['gpu_milli'] > 0 and spec['gpu_milli'] == 0:
        return "Not Supported"
    elif row['gpu_milli'] > 0 and spec['gpu_milli'] > 0:
        gpu = row['gpu_milli'] / spec['gpu_milli']
    else:
        gpu = 0
    max_val = max(cpu, mem, gpu)
    return int(np.ceil(max_val)) if np.isfinite(max_val) else "Not Supported"



for name, spec in AWS_INSTANCES.items():
    df[f"{name}_needed"] = df.apply(lambda row: estimate_aws_servers(row, spec), axis=1)


cols = ["name", "cpu_milli", "memory_mib", "gpu_milli", "servers_needed"] + [f"{name}_needed" for name in AWS_INSTANCES]
print("\nEstimated AWS Servers Needed per Pod:")
print(df[cols].head(10)) 

# Optional: Save to CSV
# df.to_csv("pods_with_aws_estimate.csv", index=False)
