# Churn Prediction Project

Mục tiêu của dự án là xây dựng mô hình học máy để dự đoán khách hàng có nguy cơ rời bỏ (churn) dựa trên dữ liệu khách hàng và hành vi mua sắm.

## 1. Cấu trúc thư mục

- `data/raw/`: dữ liệu gốc từ file Excel/CSV
- `data/processed/`: dữ liệu đã được làm sạch và chuẩn hóa để train/test
- `notebooks/`: các notebook phân tích dữ liệu, feature engineering, train và evaluate
- `models/`: mô hình đã được huấn luyện

## 2. Dataset hiện có

Dataset đã được tải và kiểm tra với 5,630 dòng và 20 cột, trong đó cột `Churn` là nhãn mục tiêu.

## 3. Mục tiêu học tập

1. Hiểu dữ liệu và phân tích EDA
2. Xử lý missing values, encoding, engineering features
3. Huấn luyện baseline model và model mạnh hơn
4. So sánh các metric phù hợp với bài toán churn
5. Lưu mô hình và triển khai demo đơn giản

## 4. Quy trình làm việc đề xuất

### Bước 1: Data Understanding
- Đọc dataset từ `data/raw/churn_prediction.xlsx`
- Kiểm tra kiểu dữ liệu, missing values, phân bố target

### Bước 2: EDA
- Phân tích tỷ lệ churn theo từng nhóm
- Vẽ biểu đồ phân phối các biến số và biến phân loại
- Xem mối quan hệ giữa biến và target

### Bước 3: Feature Engineering
- Điền missing values
- Encode biến phân loại
- Tạo feature mới nếu cần
- Chia train/test

### Bước 4: Model Training
- Baseline: Logistic Regression
- So sánh: RandomForest / XGBoost / LightGBM
- Dùng `Pipeline` kèm `StandardScaler` và `SMOTE` nếu cần

### Bước 5: Model Evaluation
- Theo dõi: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Ưu tiên model có Recall cao ở lớp churn

### Bước 6: Deployment
- Lưu model vào `models/churn_pipeline.pkl`
- Demo bằng Streamlit hoặc FastAPI

## 5. Khuyến nghị model

Với dataset thương mại điện tử kiểu này, Random Forest, XGBoost hoặc LightGBM thường cho hiệu quả tốt hơn Logistic Regression. Tuy nhiên, bạn nên bắt đầu với baseline để hiểu dữ liệu và thiết lập benchmark. Ở bài này áp dụng cả Logistic Regression và Tree-base như Random Forest, XGBoost, LightGBM để so sánh hiệu quả.

## 6. Bắt đầu

- Mở thư mục `notebooks/`
- Chạy notebook `01_data_understanding.ipynb` trước
- Tiếp tục với `02_eda.ipynb`, `03_feature_engineering.ipynb`, `04_model_training.ipynb`, `05_model_evaluation.ipynb`

## 7. Metrics quan trọng

- Recall: để phát hiện khách hàng sắp rời bỏ
- Precision: độ tin cậy khi cảnh báo
- F1-score: tổng hợp precision và recall
- ROC-AUC: khả năng phân biệt hai lớp
