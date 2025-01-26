import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import shap
# ============================
# 1. 加载数据
# ============================
# 加载日数据
daily_data = pd.read_csv('daily.csv', header=None, names=['Date', 'AQI', 'TEMP'])
# 加载月数据
monthly_data = pd.read_csv('monthly.csv', header=None, names=['Date', 'Num', 'Rate'])
# 加载年数据
yearly_data = pd.read_csv('yearly.csv', header=None, names=['Year', 'Idv_Inc', 'Hh_Inc'])

# 检查数据的前几行
# print("Daily Data Head:")
# print(daily_data.head())
# print("\nMonthly Data Head:")
# print(monthly_data.head())
# print("\nYearly Data Head:")
# print(yearly_data.head())

# 清理数据，移除非日期格式的内容
daily_data = daily_data[daily_data['Date'] != 'Date']
monthly_data = monthly_data[monthly_data['Date'] != 'Date']
yearly_data = yearly_data[yearly_data['Year'] != 'Year']

# 将日期列转换为datetime类型
try:
    daily_data['Date'] = pd.to_datetime(daily_data['Date'], format='%Y/%m/%d')
    monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], format='%Y/%m')
except ValueError as e:
    print(f"Error converting dates: {e}")
    print("Checking for non-date formats in the Date column...")
    print(daily_data[daily_data['Date'].str.contains('[a-zA-Z]', na=False)])
    print(monthly_data[monthly_data['Date'].str.contains('[a-zA-Z]', na=False)])

# 确保Year列的数据类型为整数
# 移除非数值内容
yearly_data = yearly_data[yearly_data['Year'].str.contains('[a-zA-Z]', na=False) == False]
yearly_data['Year'] = yearly_data['Year'].astype(int)

# 定义训练集和测试集的时间范围
train_start = pd.Timestamp('2011-01-01')
train_end = pd.Timestamp('2021-12-31')
test_start = pd.Timestamp('2022-01-01')
test_end = pd.Timestamp('2022-12-31')

# 划分训练集和测试集
daily_train = daily_data[(daily_data['Date'] >= train_start) & (daily_data['Date'] <= train_end)]
daily_test = daily_data[(daily_data['Date'] >= test_start) & (daily_data['Date'] <= test_end)]

monthly_train = monthly_data[(monthly_data['Date'] >= train_start) & (monthly_data['Date'] <= train_end)]
monthly_test = monthly_data[(monthly_data['Date'] >= test_start) & (monthly_data['Date'] <= test_end)]

# 划分年数据训练集和测试集
yearly_train = yearly_data[(yearly_data['Year'] >= 2011) & (yearly_data['Year'] <= 2021)]
yearly_test = yearly_data[(yearly_data['Year'] >= 2022) & (yearly_data['Year'] <= 2022)]

# 确保数据为数值类型，并处理缺失值
daily_train = daily_train.dropna(subset=['AQI', 'TEMP']).astype({'AQI': float, 'TEMP': float})
daily_test = daily_test.dropna(subset=['AQI', 'TEMP']).astype({'AQI': float, 'TEMP': float})

monthly_train = monthly_train.dropna(subset=['Num', 'Rate']).astype({'Num': float, 'Rate': float})
monthly_test = monthly_test.dropna(subset=['Num', 'Rate']).astype({'Num': float, 'Rate': float})

yearly_train = yearly_train.dropna(subset=['Idv_Inc', 'Hh_Inc']).astype({'Idv_Inc': float, 'Hh_Inc': float})
yearly_test = yearly_test.dropna(subset=['Idv_Inc', 'Hh_Inc']).astype({'Idv_Inc': float, 'Hh_Inc': float})

# 归一化
scaler_daily = MinMaxScaler()
scaler_monthly = MinMaxScaler()
scaler_yearly = MinMaxScaler()


daily_train[['AQI', 'TEMP']] = scaler_daily.fit_transform(daily_train[['AQI', 'TEMP']])
daily_test[['AQI', 'TEMP']] = scaler_daily.transform(daily_test[['AQI', 'TEMP']])

monthly_train[['Num', 'Rate']] = scaler_monthly.fit_transform(monthly_train[['Num', 'Rate']])
monthly_test[['Num', 'Rate']] = scaler_monthly.transform(monthly_test[['Num', 'Rate']])

yearly_train[['Idv_Inc', 'Hh_Inc']] = scaler_yearly.fit_transform(yearly_train[['Idv_Inc', 'Hh_Inc']])
yearly_test[['Idv_Inc', 'Hh_Inc']] = scaler_yearly.transform(yearly_test[['Idv_Inc', 'Hh_Inc']])

# 将数据转换为张量
daily_train_tensor = torch.tensor(daily_train[['AQI', 'TEMP']].values, dtype=torch.float32)
daily_test_tensor = torch.tensor(daily_test[['AQI', 'TEMP']].values, dtype=torch.float32)

monthly_train_tensor = torch.tensor(monthly_train[['Num', 'Rate']].values, dtype=torch.float32)
monthly_test_tensor = torch.tensor(monthly_test[['Num', 'Rate']].values, dtype=torch.float32)

yearly_train_tensor = torch.tensor(yearly_train[['Idv_Inc', 'Hh_Inc']].values, dtype=torch.float32)
yearly_test_tensor = torch.tensor(yearly_test[['Idv_Inc', 'Hh_Inc']].values, dtype=torch.float32)


# # 打印张量形状
# print("\nDaily Train Tensor Shape:", daily_train_tensor.shape)
# print("Daily Test Tensor Shape:", daily_test_tensor.shape)
# print("Monthly Train Tensor Shape:", monthly_train_tensor.shape)
# print("Monthly Test Tensor Shape:", monthly_test_tensor.shape)
# print("Yearly Train Tensor Shape:", yearly_train_tensor.shape)
# print("Yearly Test Tensor Shape:", yearly_test_tensor.shape)

# ============================
# 2. 定义特征提取 LSTM 模型
# ============================
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入 x 的形状是 [batch_size, a, input_size]
        # LSTM 处理
        lstm_out, _ = self.lstm(x)  # LSTM 输出形状是 [batch_size, a, hidden_size]
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # 形状变为 [batch_size, hidden_size]
        
        # 通过全连接层
        features = self.fc(lstm_out)  # 形状变为 [batch_size, output_size]
        
        
        return features  # 返回 [batch_size, output_size]
# 3. 定义主模型（包含 LSTM 和 MLP 部分）
# ============================
class AnnualPredictionModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_output_size, mlp_input_size, mlp_hidden_size, mlp_output_size):
        super(AnnualPredictionModel, self).__init__()
        # LSTM 部分
        self.lstm1 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)
        self.lstm2 = LSTMFeatureExtractor(lstm_input_size, lstm_hidden_size, num_layers=2, output_size=lstm_output_size)

        # MLP 部分
        self.fc1 = nn.Linear(2 * lstm_output_size, mlp_hidden_size)  # 两个 LSTM 特征拼接
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size)

    def forward(self, daily_data, monthly_data):
        # LSTM 提取特征
        daily_features = self.lstm1(daily_data)  # [1, lstm_output_size]
        monthly_features = self.lstm2(monthly_data)  # [1, lstm_output_size]

        # 拼接特征
        combined_features = torch.cat((daily_features, monthly_features), dim=1)  # [1, 2 * lstm_output_size]

        # MLP 进行预测
        x = self.relu(self.fc1(combined_features))  # [1, mlp_hidden_size]
        x = self.fc2(x)  # [1, mlp_output_size]
      
        
#         print(f"Model Output Shape: {x.shape}")
        return x  # 返回 [1, mlp_output_size]
# ============================
# 4. 数据集准备
# ============================
class AnnualDataset(Dataset):
    def __init__(self, daily_data, monthly_data, yearly_target):
        self.daily_data = daily_data
        self.monthly_data = monthly_data
        self.yearly_target = yearly_target

    def __len__(self):
        return len(self.yearly_target)

    def __getitem__(self, idx):
        daily_seq = self.daily_data[idx]  # 日数据序列
        monthly_seq = self.monthly_data[idx]  # 月数据序列
        target = self.yearly_target[idx]  # 年目标数据
        return (daily_seq, monthly_seq), target


# 创建训练集和测试集
# 创建训练集和测试集
train_dataset = AnnualDataset(daily_train_tensor, monthly_train_tensor, yearly_train_tensor)
test_dataset = AnnualDataset(daily_test_tensor, monthly_test_tensor, yearly_test_tensor)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=len(yearly_train_tensor), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(yearly_test_tensor), shuffle=False)

# ============================
# 5. 定义并训练模型
# ============================
# 定义模型
# 定义模型
# 定义模型
# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnnualPredictionModel(
    lstm_input_size=2,
    lstm_hidden_size=64,
    lstm_output_size=64,
    mlp_input_size=128,
    mlp_hidden_size=64,
    mlp_output_size=2
).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# 训练模型
num_epochs = 1900
losses = []  # 用于存储每个 epoch 的损失

# 在训练代码中记录损失
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0  # 初始化每个 epoch 的损失
    batch_count = 0   # 记录 batch 数量

    for inputs, targets in train_loader:
        daily_data, monthly_data = inputs
        daily_data, monthly_data, targets = daily_data.to(device), monthly_data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        daily_data = daily_data.view(11, -1, 2)
        monthly_data = monthly_data.view(11, -1, 2)
        outputs = model(daily_data, monthly_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 累加 batch 的损失
        epoch_loss += loss.item()
        batch_count += 1

    # 计算每个 epoch 的平均损失
    if batch_count > 0:
        epoch_loss /= batch_count
        losses.append(epoch_loss)
    else:
        print(f"Warning: No batches processed in epoch {epoch + 1}")

    # 打印调试信息
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 检查 losses 列表是否为空
if len(losses) == 0:
    raise ValueError("Training losses are empty. Check if data is loaded correctly and loss is calculated.")

# 绘制损失曲线
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.savefig("training_loss_curve.png", bbox_inches='tight', dpi=300)
plt.close()

print("Training loss curve saved as 'training_loss_curve.png'")

# 保存模型
print(model)
torch.save(model.state_dict(), 'annual_prediction_model.pth')
print("Model saved successfully!")

# ============================
# 6. 加载模型并进行预测
# ============================
# 加载模型
# 加载模型
# 加载模型
# 加载模型
# 加载模型
model = AnnualPredictionModel(
    lstm_input_size=2,
    lstm_hidden_size=64,
    lstm_output_size=64,
    mlp_input_size=128,
    mlp_hidden_size=64,
    mlp_output_size=2
).to(device)
model.load_state_dict(torch.load('annual_prediction_model.pth'))
model.eval()
sum=0
# 使用模型进行预测
predictions = []
ground_truths = []  # 用于存储真实值
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
with torch.no_grad():
    for inputs, targets in test_loader:
        daily_data, monthly_data = inputs
        daily_data, monthly_data = daily_data.to(device), monthly_data.to(device)
        daily_data=daily_data.view(1,-1,2)
        monthly_data=monthly_data.view(1,-1,2)
        outputs = model(daily_data, monthly_data)
        while(sum<1):
            writer = SummaryWriter(log_dir='./logs')

    # 将模型结构写入 TensorBoard 日志文件
            writer.add_graph(model, (daily_data, monthly_data))
            sum=1
            # 关闭 SummaryWriter
            writer.close()

        # 将预测值和真实值存储到列表中
        predictions.append(outputs.cpu().numpy())
        ground_truths.append(targets.cpu().numpy())

# 将预测结果和真实结果拼接成一个数组
predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)

# 反归一化预测值和真实值
predictions = scaler_yearly.inverse_transform(predictions)
ground_truths = scaler_yearly.inverse_transform(ground_truths)

# 打印预测值和真实值
print("Predictions shape:", predictions.shape)
print("Ground Truths shape:", ground_truths.shape)
print("\nPredictions:")
print(predictions)
print("\nGround Truths:")
print(ground_truths)
# import torch
# import shap
# import numpy as np
# import matplotlib.pyplot as plt
# from torch import nn

# # 确保模型在评估模式
# model.eval()

# # 定义包装器函数
# class ModelWrapper(nn.Module):
#     def __init__(self, model, seq_len_daily, seq_len_monthly, daily_features, monthly_features):
#         super(ModelWrapper, self).__init__()
#         self.model = model
#         self.seq_len_daily = seq_len_daily
#         self.seq_len_monthly = seq_len_monthly
#         self.daily_features = daily_features
#         self.monthly_features = monthly_features

#     def forward(self, daily_data, monthly_data):
#         # 确保 daily_data 和 monthly_data 是张量并且在正确的设备上
#         if not isinstance(daily_data, torch.Tensor):
#             daily_data = torch.tensor(daily_data, dtype=torch.float32).to(device)  # 将数据移动到正确的设备
#         if not isinstance(monthly_data, torch.Tensor):
#             monthly_data = torch.tensor(monthly_data, dtype=torch.float32).to(device)  # 将数据移动到正确的设备
        
#         batch_size = daily_data.size(0)  # 获取 batch_size

#         # 分割数据为 daily 和 monthly 部分
#         daily_data = daily_data.view(batch_size, self.seq_len_daily, self.daily_features)
#         monthly_data = monthly_data.view(batch_size, self.seq_len_monthly, self.monthly_features)
        
#         return self.model(daily_data, monthly_data)

# # 从训练集中选择背景数据
# background_data = next(iter(train_loader))
# background_daily, background_monthly = background_data[0]
# background_daily = background_daily.to(device)
# background_monthly = background_monthly.to(device)

# # 打印背景数据的形状以调试
# print(f"background_daily shape: {background_daily.shape}")
# print(f"background_monthly shape: {background_monthly.shape}")

# # 假设没有明确的时间序列结构，我们将数据扩展为适当的 seq_len 和 features
# # 将数据调整为 (batch_size, seq_len, features)
# seq_len_daily = 1
# seq_len_monthly = 1
# daily_features = background_daily.size(1)  # 特征数量（在这个例子中是 2）
# monthly_features = background_monthly.size(1)  # 特征数量（在这个例子中是 2）

# # 将数据转换为 3D 形状 (batch_size, seq_len, features)
# background_daily = background_daily.view(background_daily.size(0), seq_len_daily, daily_features)
# background_monthly = background_monthly.view(background_monthly.size(0), seq_len_monthly, monthly_features)

# # 将背景数据合并为一个二维数组
# background_data_combined = np.concatenate((
#     background_daily.view(background_daily.size(0), -1).cpu().numpy(),
#     background_monthly.view(background_monthly.size(0), -1).cpu().numpy()
# ), axis=1)

# # 包装模型
# wrapped_model = ModelWrapper(model, seq_len_daily, seq_len_monthly, daily_features, monthly_features).to(device)

# # 使用 DeepExplainer
# # 注意：传递给 DeepExplainer 的背景数据应分别为 daily 和 monthly 数据
# explainer = shap.DeepExplainer(wrapped_model, [background_daily, background_monthly])

# # 准备测试数据
# test_data = next(iter(test_loader))
# test_daily, test_monthly = test_data[0]
# test_daily = test_daily.to(device)
# test_monthly = test_monthly.to(device)

# # 打印测试数据的形状以调试
# print(f"test_daily shape: {test_daily.shape}")
# print(f"test_monthly shape: {test_monthly.shape}")

# # 将测试数据转换为 3D 形状 (batch_size, seq_len, features)
# test_daily = test_daily.view(test_daily.size(0), seq_len_daily, daily_features)
# test_monthly = test_monthly.view(test_monthly.size(0), seq_len_monthly, monthly_features)

# # 将测试数据合并为一个二维数组
# test_data_combined = np.concatenate((
#     test_daily.view(test_daily.size(0), -1).cpu().numpy(),
#     test_monthly.view(test_monthly.size(0), -1).cpu().numpy()
# ), axis=1)

# # 计算 SHAP 值
# shap_values = explainer.shap_values([test_daily, test_monthly])  # 保持为 PyTorch 张量

# # 将 SHAP 值和测试数据拼接在一起
# shap_values_combined = shap_values
# test_data_combined = np.concatenate([test_daily.view(test_daily.size(0), -1).cpu().numpy(), 
#                                      test_monthly.view(test_monthly.size(0), -1).cpu().numpy()], axis=1)

# # 汇总图：显示每个特征的 SHAP 值
# shap.summary_plot(shap_values_combined, test_data_combined, feature_names=["Daily TEMP", "Daily AQI", "Monthly Num", "Monthly Rate"])
# plt.savefig('shap_summary_plot.png')  # 保存汇总图

# # 特定特征的依赖图：显示 "Daily TEMP" 特征的 SHAP 值
# shap.dependence_plot(0, shap_values_combined, test_data_combined, feature_names=["Daily TEMP", "Daily AQI", "Monthly Num", "Monthly Rate"])
# plt.savefig('shap_dependence_plot_daily_temp.png')  # 保存依赖图
