import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đọc dữ liệu từ file
file_path = "FINAL_USO.csv"  # Đảm bảo file này nằm trong thư mục chạy code

df = pd.read_csv(file_path)

# Chọn các cột liên quan đến giá vàng và các yếu tố ảnh hưởng
selected_columns = ['EU_Price', 'OF_Price', 'OS_Price', 'SF_Price', 'USB_Price',
                    'PLT_Price', 'PLD_Price', 'RHO_PRICE', 'USDI_Price']

df = df[selected_columns].dropna()  # Loại bỏ dòng có giá trị thiếu

# Chia dữ liệu thành tập train/test
X = df.drop(columns=['EU_Price'])  # Dữ liệu đầu vào (các yếu tố ảnh hưởng)
y = df['EU_Price']  # Mục tiêu là giá vàng (EU_Price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Lưu mô hình để sử dụng sau này
joblib.dump(model, "gold_price_model.pkl")
print("Mô hình đã được lưu thành công!")
