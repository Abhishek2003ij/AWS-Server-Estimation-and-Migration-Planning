AWS-Server-Estimation-and-Migration-Planning
This project leverages Machine Learning models (Random Forest & XGBoost) to predict the number of AWS cloud servers required to handle the same workload as traditional on-prem servers. It's built to help businesses optimize cloud resource allocation during migration or scaling.

Features
Predicts required server count based on CPU, memory, and GPU usage.

Compares traditional server load with various AWS instance types.

Uses Random Forest Regressor and XGBoost for accurate ML predictions.

Handles GPU workload requirements intelligently.

Outputs results in a human-readable format (CSV or table).

Models Used
RandomForestRegressor (Scikit-learn)

XGBRegressor (XGBoost)

Input Features
Feature	Description
cpu_milli	CPU usage in millicores
memory_mib	Memory usage in MiB
gpu_milli	GPU usage in millicores (optional)
num_gpu	Number of GPUs used

AWS Instances Compared
Instance	vCPU	Memory (MiB)	GPU
t3.large	2	8192	0
m5.2xlarge	8	32768	0
c5.4xlarge	16	32768	0
g4dn.4xlarge	16	65536	1
g4dn.12xlarge	48	196608	4

GPU instances are recommended for workloads with high gpu_milli requirements.

Sample Output

| name          | cpu_milli | memory_mib | gpu_milli | servers_needed | g4dn.4xlarge_needed |
|---------------|-----------|------------|-----------|----------------|---------------------|
| openb-pod-0000| 12000     | 16384      | 1000      | 1              | 1                   |


Collaboration & Credits
Built by Abhishek IJ — Open to collaboration and business inquiries.

License
MIT License — feel free to use, modify, and contribute.

