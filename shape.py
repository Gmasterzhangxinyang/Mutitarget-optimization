import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义LSTM特征提取器
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        features = self.fc(lstm_out)
        return features

# 定义主模型
class AnnualPredictionModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_output_size, mlp_input_size, mlp_hidden_size, mlp_output_size):
        super(AnnualPredictionModel, self).__init__()
        self.lstm1 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)
        self.lstm2 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)
        self.fc1 = nn.Linear(2 * lstm_output_size, mlp_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size)

    def forward(self, daily_data, monthly_data):
        daily_features = self.lstm1(daily_data)
        monthly_features = self.lstm2(monthly_data)
        combined_features = torch.cat((daily_features, monthly_features), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

# 初始化模型并加载权重
model = AnnualPredictionModel(
    lstm_input_size=2,
    lstm_hidden_size=64,
    lstm_output_size=64,
    mlp_input_size=128,
    mlp_hidden_size=64,
    mlp_output_size=2
).to(device)

model.load_state_dict(torch.load('45,92.pth', map_location=device))
model.eval()

# 定义优化问题
class MyProblem(ElementwiseProblem):
    def __init__(self, model):
        super().__init__(n_var=4, n_obj=2, xl=[0, 1, 14155, 2.1], xu=[100, 71, 17632, 10.9])
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x).view(1, 4).float().to(device)
        input1 = x_tensor[:, :2].view(1, 1, 2)
        input2 = x_tensor[:, 2:].view(1, 1, 2)
        output = self.model(input1, input2)
       

        # 使用scaler_yearly进行反归一化
       
        out["F"] = -output.detach().cpu().numpy().flatten()  # 取负数以实现最小化

# 创建问题实例
problem = MyProblem(model=model)

# 随机生成10000个样本点
num_samples = 10000
random_samples = np.random.uniform(low=problem.xl, high=problem.xu, size=(num_samples, problem.n_var))

# 计算随机点的目标值
random_obj_values = []
random_inputs = []
for sample in random_samples:
    sample_tensor = torch.tensor(sample).view(1, 4).float().to(device)
    input1 = sample_tensor[:, :2].view(1, 1, 2)
    input2 = sample_tensor[:, 2:].view(1, 1, 2)
    output = model(input1, input2).detach().cpu().numpy().flatten()
    random_obj_values.append(-output)  # 取负数以实现最小化
    random_inputs.append(sample)
random_obj_values = np.array(random_obj_values)
random_inputs = np.array(random_inputs)
# 定义一个函数来计算Pareto前沿
def compute_pareto_front(obj_values, inputs):
    is_efficient = np.ones(obj_values.shape[0], dtype=bool)
    for i, c in enumerate(obj_values):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(obj_values[is_efficient] < c, axis=1)  # 保留所有不被支配的解
            is_efficient[i] = True
    return obj_values[is_efficient], inputs[is_efficient]

# 计算Pareto前沿
pareto_front, pareto_inputs = compute_pareto_front(random_obj_values, random_inputs)

# 打印Pareto前沿点的数量和对应的输入变量
print(f"Pareto前沿点的数量: {len(pareto_front)}")
print("Pareto前沿点对应的输入变量:")
print(pareto_inputs)


# 绘制10000个随机样本点和Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(random_obj_values[:, 0], random_obj_values[:, 1], c='gray', marker='o', s=10, alpha=0.5, label='Random Samples')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue', marker='o', s=50, label='Pareto Front')
plt.xlabel('Personal Income (Negative)')
plt.ylabel('Family Income (Negative)')
plt.title('Pareto Front in Objective Space')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('muti——low.png')

# 将Pareto前沿点对应的输入变量保存为CSV文件
columns = ['AQI', 'TEMP', 'Employ_num', 'Unemployment_rate']
pareto_inputs_df = pd.DataFrame(pareto_inputs, columns=columns)
pareto_inputs_df.to_csv('pareto_front_inputs.csv', index=False)
print("Pareto前沿点对应的输入变量已保存到 pareto_front_inputs.csv 文件中。")

#以下为还原成最大值

# import pandas as pd
# import torch
# from sklearn.preprocessing import MinMaxScaler
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.optimize import minimize
# from pymoo.core.problem import ElementwiseProblem
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 定义LSTM特征提取器
# class LSTMFeatureExtractor(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMFeatureExtractor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]
#         features = self.fc(lstm_out)
#         return features

# # 定义主模型
# class AnnualPredictionModel(nn.Module):
#     def __init__(self, lstm_input_size, lstm_hidden_size, lstm_output_size, mlp_input_size, mlp_hidden_size, mlp_output_size):
#         super(AnnualPredictionModel, self).__init__()
#         self.lstm1 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)
#         self.lstm2 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)
#         self.fc1 = nn.Linear(2 * lstm_output_size, mlp_hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size)

#     def forward(self, daily_data, monthly_data):
#         daily_features = self.lstm1(daily_data)
#         monthly_features = self.lstm2(monthly_data)
#         combined_features = torch.cat((daily_features, monthly_features), dim=1)
#         x = self.relu(self.fc1(combined_features))
#         x = self.fc2(x)
#         return x

# # 初始化模型并加载权重
# model = AnnualPredictionModel(
#     lstm_input_size=2,
#     lstm_hidden_size=64,
#     lstm_output_size=64,
#     mlp_input_size=128,
#     mlp_hidden_size=64,
#     mlp_output_size=2
# ).to(device)

# model.load_state_dict(torch.load('45,92.pth', map_location=device))
# model.eval()

# # 定义优化问题
# class MyProblem(ElementwiseProblem):
#     def __init__(self, model):
#         super().__init__(n_var=4, n_obj=2, xl=[0, 1, 14155, 2.1], xu=[100, 71, 17632, 10.9])
#         self.model = model

#     def _evaluate(self, x, out, *args, **kwargs):
#         x_tensor = torch.tensor(x).view(1, 4).float().to(device)
#         input1 = x_tensor[:, :2].view(1, 1, 2)
#         input2 = x_tensor[:, 2:].view(1, 1, 2)
#         output = self.model(input1, input2)
#         out["F"] = -output.detach().cpu().numpy().flatten()  # 取负数以实现最小化

# # 创建问题实例
# problem = MyProblem(model=model)

# # 随机生成10000个样本点
# num_samples = 10000
# random_samples = np.random.uniform(low=problem.xl, high=problem.xu, size=(num_samples, problem.n_var))

# # 计算随机点的目标值
# random_obj_values = []
# random_inputs = []
# for sample in random_samples:
#     sample_tensor = torch.tensor(sample).view(1, 4).float().to(device)
#     input1 = sample_tensor[:, :2].view(1, 1, 2)
#     input2 = sample_tensor[:, 2:].view(1, 1, 2)
#     output = model(input1, input2).detach().cpu().numpy().flatten()
#     random_obj_values.append(-output)  # 取负数以实现最小化
#     random_inputs.append(sample)
# random_obj_values = np.array(random_obj_values)
# random_inputs = np.array(random_inputs)

# # 定义一个函数来计算Pareto前沿
# def compute_pareto_front(obj_values, inputs):
#     is_efficient = np.ones(obj_values.shape[0], dtype=bool)
#     for i, c in enumerate(obj_values):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(obj_values[is_efficient] < c, axis=1)  # 保留所有不被支配的解
#             is_efficient[i] = True
#     return obj_values[is_efficient], inputs[is_efficient]

# # 计算Pareto前沿
# pareto_front, pareto_inputs = compute_pareto_front(random_obj_values, random_inputs)

# # 打印Pareto前沿点的数量和对应的输入变量
# print(f"Pareto前沿点的数量: {len(pareto_front)}")
# print("Pareto前沿点对应的输入变量:")
# print(pareto_inputs)

# # 将目标函数值还原回原始的正数形式
# original_pareto_front = -pareto_front
# original_random_obj_values = -random_obj_values

# # 绘制10000个随机样本点和Pareto前沿
# plt.figure(figsize=(10, 6))
# plt.scatter(original_random_obj_values[:, 0], original_random_obj_values[:, 1], c='gray', marker='o', s=10, alpha=0.5, label='Random Samples')
# plt.scatter(original_pareto_front[:, 0], original_pareto_front[:, 1], c='blue', marker='o', s=50, label='Pareto Front')
# plt.xlabel('Personal Income')
# plt.ylabel('Family Income')
# plt.title('Pareto Front in Objective Space')
# plt.legend()
# plt.grid(True)
# plt.show()


# # 将Pareto前沿点对应的输入变量保存为CSV文件
# columns = ['AQI', 'TEMP', 'Employ_num', 'Unemployment_rate']
# pareto_inputs_df = pd.DataFrame(pareto_inputs, columns=columns)
# pareto_inputs_df.to_csv('pareto_front_inputs.csv', index=False)
# print("Pareto前沿点对应的输入变量已保存到 pareto_front_inputs_low.csv 文件中。")




