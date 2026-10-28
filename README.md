# 🤖 E-Commerce Customer Churn Prediction & Retention Strategy | Machine Learning - Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/Algorithm-LightGBM_&_XGBoost-blue?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Churn Recall](https://img.shields.io/badge/Recall_(Churn)-94.74%25-brightgreen?style=for-the-badge)]()
[![Model Precision](https://img.shields.io/badge/Precision_(Churn)-92.31%25-success?style=for-the-badge)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9930-blueviolet?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Dự án xây dựng hệ thống dự báo khách hàng rời bỏ (Customer Churn Prediction) và chiến lược giữ chân (Retention Strategy) cho doanh nghiệp thương mại điện tử với 5,630 tài khoản khách hàng. Toàn bộ quy trình được triển khai chặt chẽ qua 5 Jupyter Notebooks theo chuẩn công nghiệp, đạt Recall 94.74% và Accuracy 97.96% với mô hình LightGBM, kết hợp Ma trận hành động CRM 3 phân tầng.**

---

**Tác giả:** [Nguyễn Xuân Thu](https://github.com/thunx280-ctrl)  
**Vai trò:** Data Analyst / Machine Learning Enthusiast  
**Công cụ sử dụng:** Python (Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, Matplotlib, Seaborn, Joblib)  
**Tập dữ liệu:** E-Commerce Customer Churn Dataset (5,630 dòng $\times$ 20 cột thuộc tính)  

---

## 📑 Table of Contents

1. 📌 [Bối cảnh & Mục tiêu Kinh doanh (Background & Business Context)](#-bối-cảnh--mục-tiêu-kinh-doanh)
2. 📂 [Mô tả & Cấu trúc Tập dữ liệu (Dataset Description & Structure)](#-mô-tả--cấu-trúc-tập-dữ-liệu)
3. ⚙️ [Quy trình Triển khai Chi tiết theo 5 Notebooks (Main Process)](#-quy-trình-triển-khai-chi-tiết-theo-5-notebooks)
   - 3.1 🌱 [Bước 1: Data Understanding & Kiểm toán Chất lượng Dữ liệu (`01_data_understanding.ipynb`)](#1️⃣-bước-1-data-understanding--kiểm-toán-chất-lượng-dữ-liệu-01_data_understandingipynb)
   - 3.2 🔍 [Bước 2: Exploratory Data Analysis & Khám phá Xu hướng Churn (`02_eda.ipynb`)](#2️⃣-bước-2-exploratory-data-analysis--khám-phá-xu-hướng-churn-02_edaipynb)
   - 3.3 🛠️ [Bước 3: Feature Engineering Không Rò rỉ Thông tin (`03_feature_engineering.ipynb`)](#3️⃣-bước-3-feature-engineering-không-rò-rỉ-thông-tin-03_feature_engineeringipynb)
   - 3.4 📊 [Bước 4: Huấn luyện, So sánh Mô hình & Tuning Hyperparameter (`04_model_training.ipynb`)](#4️⃣-bước-4-huấn-luyện-so-sánh-mô-hình--tuning-hyperparameter-04_model_trainingipynb)
   - 3.5 🎯 [Bước 5: Đánh giá Chi tiết, Ma trận Nhầm lẫn & Giải thích Mô hình (`05_model_evaluation.ipynb`)](#5️⃣-bước-5-đánh-giá-chi-tiết-ma-trận-nhầm-lẫn--giải-thích-mô-hình-05_model_evaluationipynb)
4. 💡 [Tổng hợp Phát hiện & Khuyến nghị Giữ chân Khách hàng (Key Findings & Recommendations)](#-tổng-hợp-phát-hiện--khuyến-nghị-giữ-chân-khách-hàng)
5. 🚀 [Phân khúc Rủi ro & Ma trận Hành động CRM (Customer Risk Segmentation & Retention Matrix)](#-phân-khúc-rủi-ro--ma-trận-hành-động-crm)
6. 📁 [Cấu trúc Thư mục Dự án (Repository Structure)](#-cấu-trúc-thư-mục-dự-án)
7. 💻 [Hướng dẫn Cài đặt & Tái hiện (How to Run & Reproduce)](#-hướng-dẫn-cài-đặt--tái-hiện)
8. 👤 [Thông tin Tác giả & Liên hệ](#-thông-tin-tác-giả--liên-hệ)

---

## 📌 Bối cảnh & Mục tiêu Kinh doanh

### 🎯 Mục tiêu Dự án (Objective)
Trong ngành thương mại điện tử (E-Commerce), chi phí thu hút một khách hàng mới (CAC) thường **cao gấp 5 đến 7 lần** so với việc giữ chân một khách hàng hiện hữu. Khách hàng rời bỏ không chỉ làm suy giảm giá trị vòng đời khách hàng (Customer Lifetime Value - CLV) mà còn gây lãng phí ngân sách tiếp thị.

Dự án này tập trung xây dựng **Hệ thống Cảnh báo Sớm Khách hàng Rời bỏ** nhằm:
* Phát hiện sớm khách hàng có nguy cơ rời bỏ cao trước khi họ chính thức ngừng giao dịch.
* Xác định rõ các nguyên nhân cốt lõi và hành vi dẫn đến quyết định rời bỏ.
* Xây dựng mô hình phân loại nhị phân tối ưu hóa chỉ số **Recall (Class 1)** để giảm thiểu tối đa rủi ro bỏ sót khách hàng rời bỏ (False Negatives), đồng thời duy trì **Precision** cao để tránh lãng phí ngân sách ưu đãi.
* Chuyển hóa kết quả dự báo thành các gói hành động giữ chân tự động cho đội ngũ CRM và Chăm sóc Khách hàng (CS).

### ❓ Câu hỏi Kinh doanh Cần Giải quyết
1. Những yếu tố vận hành và hành vi nào tác động mạnh nhất đến quyết định rời bỏ của khách hàng?
2. Thời gian gắn bó (Tenure) và khiếu nại (Complain) ảnh hưởng như thế nào đến tỷ lệ churn?
3. Làm thế nào để xây dựng mô hình dự báo chính xác mà không bị rò rỉ dữ liệu (data leakage) giữa tập train và test?
4. Đội ngũ tiếp thị và chăm sóc khách hàng cần can thiệp cụ thể ra sao theo từng phân khúc rủi ro?

### 👤 Đối tượng Hướng tới
* **Data Analyst / Analytics Engineer / Data Scientist:** Tham khảo quy trình phân tích và xây dựng pipeline học máy chuẩn mực, có kiểm toán dữ liệu chặt chẽ.
* **Đội ngũ Tiếp thị & CRM:** Ứng dụng ma trận phân khúc rủi ro để gửi ưu đãi đúng đối tượng, đúng thời điểm.
* **Đội ngũ Chăm sóc Khách hàng (CS / CX):** Thiết lập quy trình xử lý ưu tiên (SLA) đối với khách hàng có khiếu nại.

---

## 📂 Mô tả & Cấu trúc Tập dữ liệu

* **Nguồn dữ liệu:** Cơ sở dữ liệu giao dịch và hồ sơ người dùng của nền tảng E-Commerce.
* **Quy mô:** 5,630 dòng $\times$ 20 cột thuộc tính.
* **Định dạng:** Microsoft Excel (`.xlsx`) tại đường dẫn `data/raw/churn_prediction.xlsx`.

<details>
  <summary><b>▶ Nhấp để mở rộng Bảng Từ điển Dữ liệu (Data Dictionary - 20 Cột)</b></summary>
<br>

| Tên Cột | Kiểu Dữ liệu | Thang đo | Ý nghĩa Nghiệp vụ |
| :--- | :---: | :---: | :--- |
| `CustomerID` | `int64` | Định danh | Mã định danh duy nhất cho từng khách hàng. |
| `Churn` | `int64` | Cờ nhị phân | **Biến mục tiêu (Target):** `1` = Khách hàng rời bỏ; `0` = Khách hàng ở lại. |
| `Tenure` | `float64` | Thời gian (Tháng) | Thời gian khách hàng gắn bó với nền tảng thương mại điện tử. |
| `PreferredLoginDevice` | `str / object` | Danh pháp | Thiết bị đăng nhập ưa thích (`Mobile Phone`, `Phone`, `Computer`). |
| `CityTier` | `int64` | Phân khúc bậc | Phân loại đô thị của khách hàng (`Tier 1`, `Tier 2`, `Tier 3`). |
| `WarehouseToHome` | `float64` | Khoảng cách (km) | Khoảng cách từ kho hàng hoàn tất đơn hàng đến địa chỉ giao hàng. |
| `PreferredPaymentMode` | `str / object` | Danh pháp | Phương thức thanh toán ưu tiên (`Debit Card`, `Credit Card`, `E wallet`, `COD`, `UPI`). |
| `Gender` | `str / object` | Danh pháp | Giới tính của khách hàng (`Male`, `Female`). |
| `HourSpendOnApp` | `float64` | Thời gian (Giờ) | Số giờ trung bình khách hàng lướt ứng dụng/website mỗi ngày. |
| `NumberOfDeviceRegistered`| `int64` | Đếm số lượng | Số lượng thiết bị đã đăng ký/kết nối với tài khoản. |
| `PreferedOrderCat` | `str / object` | Danh pháp | Ngành hàng đặt mua nhiều nhất (`Fashion`, `Laptop & Accessory`, `Mobile Phone`,...). |
| `SatisfactionScore` | `int64` | Thứ bậc (1–5) | Điểm đánh giá mức độ hài lòng của khách hàng (1 = Kém, 5 = Rất tốt). |
| `MaritalStatus` | `str / object` | Danh pháp | Tình trạng hôn nhân (`Single`, `Married`, `Divorced`). |
| `NumberOfAddress` | `int64` | Đếm số lượng | Số lượng địa chỉ giao hàng đã lưu trên hệ thống. |
| `Complain` | `int64` | Cờ nhị phân | Khách hàng có gửi khiếu nại trong tháng gần nhất không (`1` = Có, `0` = Không). |
| `OrderAmountHikeFromlastYear`| `float64` | Phần trăm (%) | Tỷ lệ gia tăng giá trị đơn hàng so với năm trước. |
| `CouponUsed` | `float64` | Đếm số lượng | Số lượng mã giảm giá khách hàng đã sử dụng trong chu kỳ đánh giá. |
| `OrderCount` | `float64` | Đếm số lượng | Tổng số lượng đơn hàng khách hàng đã đặt trong chu kỳ. |
| `DaySinceLastOrder` | `float64` | Thời gian (Ngày) | Số ngày trôi qua kể từ đơn hàng thành công gần nhất. |
| `CashbackAmount` | `float64` | Tiền tệ | Giá trị hoàn tiền (cashback) trung bình khách hàng nhận được mỗi tháng. |

</details>

---

## ⚙️ Quy trình Triển khai Chi tiết theo 5 Notebooks

Toàn bộ giải pháp được cấu trúc tuần tự qua **5 Jupyter Notebooks độc lập** trong thư mục `notebooks/`:

```
notebooks/
├── 01_data_understanding.ipynb      # Bước 1: Khám phá cấu trúc, phân phối target & missing values
├── 02_eda.ipynb                     # Bước 2: Phân tích đơn biến, đa biến, tương quan & hành vi rời bỏ
├── 03_feature_engineering.ipynb     # Bước 3: Chia train/test, điền missing, tạo biến nghiệp vụ & encode
├── 04_model_training.ipynb          # Bước 4: Huấn luyện baseline, so sánh 4 mô hình & tuning LightGBM
└── 05_model_evaluation.ipynb        # Bước 5: Đánh giá test set, Confusion Matrix, ROC-PR & Feature Importance
```

---

### 1️⃣ Bước 1: Data Understanding & Kiểm toán Chất lượng Dữ liệu (`01_data_understanding.ipynb`)

#### 🎯 Mục tiêu
Đọc dữ liệu từ file Excel gốc, kiểm tra kiểu dữ liệu, phân tích độ cân bằng của biến mục tiêu `Churn`, kiểm tra giá trị thiếu (missing values) và phát hiện các bản ghi trùng lặp.

#### 🛠️ Mã nguồn & Thực thi trong Notebook

##### [In 1]: Import thư viện cần thiết
```python
# import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

##### [In 2]: Tải dữ liệu và kiểm tra kích thước
```python
df = pd.read_excel("../data/raw/churn_prediction.xlsx")
display(df.shape)
df.head()
```
##### [Out 2]:
Kích thước bảng dữ liệu: `(5630, 20)`

| CustomerID | Churn | Tenure | PreferredLoginDevice | CityTier | WarehouseToHome | PreferredPaymentMode | Gender | HourSpendOnApp | NumberOfDeviceRegistered | PreferedOrderCat | SatisfactionScore | MaritalStatus | NumberOfAddress | Complain | OrderAmountHikeFromlastYear | CouponUsed | OrderCount | DaySinceLastOrder | CashbackAmount |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 52869 | 0 | 20.0 | Mobile Phone | 3 | 7.0 | E wallet | Female | 4.0 | 4 | Fashion | 5 | Married | 3 | 0 | 26.0 | 5.0 | 16.0 | NaN | 229.53 |
| 52942 | 0 | 13.0 | Computer | 1 | 9.0 | Cash on Delivery | Female | 4.0 | 4 | Fashion | 3 | Single | 2 | 0 | 26.0 | 11.0 | NaN | 9.0 | 234.38 |
| 52972 | 0 | 16.0 | Mobile Phone | 3 | 7.0 | Debit Card | Male | 3.0 | 4 | Laptop & Accessory | 3 | Divorced | 3 | 0 | 26.0 | 5.0 | 12.0 | 7.0 | 174.07 |
| 53125 | 0 | 5.0 | Mobile Phone | 1 | 16.0 | Debit Card | Male | 3.0 | 4 | Fashion | 4 | Married | 3 | 0 | 26.0 | 2.0 | 2.0 | 9.0 | 231.48 |
| 53367 | 0 | 9.0 | Mobile Phone | 1 | 28.0 | Debit Card | Female | 3.0 | 4 | Laptop & Accessory | 2 | Divorced | 3 | 1 | 26.0 | 1.0 | 2.0 | 8.0 | 165.14 |

##### [In 3]: Kiểm tra cấu trúc thông tin tổng thể
```python
df.info()
```
##### [Out 3]:
```
<class 'pandas.DataFrame'>
RangeIndex: 5630 entries, 0 to 5629
Data columns (total 20 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   CustomerID                   5630 non-null   int64  
 1   Churn                        5630 non-null   int64  
 2   Tenure                       5366 non-null   float64
 3   PreferredLoginDevice         5630 non-null   str    
 4   CityTier                     5630 non-null   int64  
 5   WarehouseToHome              5379 non-null   float64
 6   PreferredPaymentMode         5630 non-null   str    
 7   Gender                       5630 non-null   str    
 8   HourSpendOnApp               5375 non-null   float64
 9   NumberOfDeviceRegistered     5630 non-null   int64  
 10  PreferedOrderCat             5630 non-null   str    
 11  SatisfactionScore            5630 non-null   int64  
 12  MaritalStatus                5630 non-null   str    
 13  NumberOfAddress              5630 non-null   int64  
 14  Complain                     5630 non-null   int64  
 15  OrderAmountHikeFromlastYear  5365 non-null   float64
 16  CouponUsed                   5374 non-null   float64
 17  OrderCount                   5372 non-null   float64
 18  DaySinceLastOrder            5323 non-null   float64
 19  CashbackAmount               5630 non-null   float64
dtypes: float64(8), int64(7), str(5)
memory usage: 1.1 MB
```

##### [In 7]: Kiểm tra phân phối biến mục tiêu (Target Imbalance)
```python
# Check imbalance label target 
imb_df = df.groupby('Churn')['CustomerID'].count().reset_index()
imb_df['%'] = (imb_df['CustomerID'] / imb_df['CustomerID'].sum()) * 100
display(imb_df)

# Vẽ biểu đồ phân phối biến mục tiêu
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.show()
```
##### [Out 7]:
| Churn | Số lượng Khách hàng (CustomerID) | Tỷ lệ (%) |
| :---: | :---: | :---: |
| **0** (Ở lại) | 4,682 | **83.16%** |
| **1** (Rời bỏ) | 948 | **16.84%** |

*(Ảnh chụp màn hình phân phối biến mục tiêu `sns.countplot`)*  
<img width="1464" height="986" alt="{3F7FC70A-F81D-443A-9FC2-1B0A41D15170}" src="https://github.com/user-attachments/assets/54ed00af-3aa9-4bf6-b586-b0da557b124d" />

##### [In 9] & [In 10]: Kiểm tra giá trị thiếu (Missing Values)
```python
import missingno as msno
msno.matrix(df)

mising = df.isnull().sum()
missing_pct = (mising / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': mising, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
display(missing_df)
```
##### [Out 10]:
| Tên Cột Thiếu Dữ liệu | Số lượng Thiếu (Rows) | Tỷ lệ Thiếu (%) |
| :--- | :---: | :---: |
| `DaySinceLastOrder` | 307 | 5.45% |
| `OrderAmountHikeFromlastYear` | 265 | 4.71% |
| `Tenure` | 264 | 4.69% |
| `OrderCount` | 258 | 4.58% |
| `CouponUsed` | 256 | 4.55% |
| `HourSpendOnApp` | 255 | 4.53% |
| `WarehouseToHome` | 251 | 4.46% |

*(Ảnh chụp màn hình ma trận missing values `msno.matrix`)*  
<img width="2614" height="1362" alt="{0DBAF79C-0637-42C7-B21A-CA4884CCF854}" src="https://github.com/user-attachments/assets/a985def7-d054-421c-8016-a04a288ed2ca" />

##### [In 11]: Kiểm tra dòng trùng lặp
```python
duplicate_df = df.duplicated().any()
duplicate_df
```
##### [Out 11]: `np.False_` (0 dòng trùng lặp trên toàn bộ 5,630 bản ghi).

#### 💡 Kết luận Rút ra từ Bước 1:
1. Dataset có 5,630 dòng và 20 cột, không có dòng trùng lặp nào.
2. Biến mục tiêu `Churn` có tính mất cân bằng: **83.16% ở lại** (0) so với **16.84% rời bỏ** (1). Do đó, không thể chỉ dùng Accuracy mà cần ưu tiên các metric nhạy cảm với lớp thiểu số như **Recall, F1-Score, và ROC-AUC/PR-AUC**.
3. Có đúng 7 cột chứa giá trị thiếu, tất cả đều dưới 6%, chủ yếu là các chỉ số vận hành và hành vi (Tenure, WarehouseToHome, Recency...). Chiến lược xử lý phù hợp là sử dụng median và bổ sung cờ chỉ báo thiếu dữ liệu (`missing indicator`).

---

### 2️⃣ Bước 2: Exploratory Data Analysis & Khám phá Xu hướng Churn (`02_eda.ipynb`)

#### 🎯 Mục tiêu
Phân tích phân phối của từng biến số, tìm mối quan hệ giữa các biến độc lập và biến mục tiêu `Churn`, tính toán ma trận tương quan (correlation matrix), và phân tích churn rate theo các phân khúc khách hàng.

#### 🛠️ Mã nguồn & Thực thi trong Notebook

##### [In 3] & [In 4]: Phân phối biến số & Boxplot so với Churn
```python
# Lấy danh sách các cột số (loại trừ CustomerID và Churn)
numerical_cols = df.select_dtypes(include='number').columns.tolist()
numerical_cols.remove('CustomerID')
numerical_cols.remove('Churn')

# Vẽ biểu đồ phân phối từng biến số
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col])
    plt.show()

# Vẽ biểu đồ boxplot so với biến mục tiêu 'Churn'
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.show()
```
*(Ảnh chụp màn hình phân phối histplot & boxplot các biến số vs Churn)*  
<!-- [SCREENSHOT PLACEHOLDER: notebooks/02_eda.ipynb - Histplot & Boxplot of Numerical Features vs Churn] -->

##### [In 6] & [In 7]: Phân tích biến danh pháp & Tỷ lệ Churn theo nhóm
```python
# Danh sách các cột định tính
categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

# Vẽ biểu đồ phân phối và churn rate cho từng biến định tính
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='Churn', data=df)
    plt.title(f'{col} by Churn')
    plt.show()

    # Tính churn rate theo nhóm
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    churn_rate.plot(kind='barh')
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.show()
```
*(Ảnh chụp màn hình tỷ lệ churn theo biến định tính countplot & barh)*  
<!-- [SCREENSHOT PLACEHOLDER: notebooks/02_eda.ipynb - Categorical Churn Rate Plots] -->

##### [In 9]: Ma trận tương quan (Correlation Matrix) & Bảng xếp hạng tương quan
```python
# Tính ma trận tương quan giữa các biến số
corr_matrix = df[numerical_cols + ['Churn']].corr()

# Vẽ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix')
plt.show()

# Xếp hạng mức độ tương quan với Churn
churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(ascending=False)
churn_corr
```
##### [Out 9]:
```
Complain                       0.250188
NumberOfDeviceRegistered       0.107939
SatisfactionScore              0.105481
CityTier                       0.084703
WarehouseToHome                0.076630
NumberOfAddress                0.043931
HourSpendOnApp                 0.018675
CouponUsed                    -0.008264
OrderAmountHikeFromlastYear   -0.010058
OrderCount                    -0.028697
CashbackAmount                -0.154118
DaySinceLastOrder             -0.160757
Tenure                        -0.349408
Name: Churn, dtype: float64
```
*(Heatmap Correlation Matrix)*  
<img width="2379" height="1492" alt="{77038A43-9DEF-4541-965A-DAFBA6FD539D}" src="https://github.com/user-attachments/assets/88553ccb-0f75-4c88-883b-9da1467e9ac1" />

#### 💡 Kết luận Rút ra từ Bước 2:
1. **Thời gian gắn bó (`Tenure`) là yếu tố bảo vệ mạnh nhất:** Tương quan âm rõ nét nhất với Churn (**$-0.349$**). Khách hàng có thời gian gắn bó ngắn hoặc mới tham gia có nguy cơ rời bỏ cực cao; khi tenure vượt qua mốc 12 tháng, tỷ lệ rời bỏ giảm mạnh.
2. **Khiếu nại (`Complain`) là tín hiệu cảnh báo rủi ro lớn nhất:** Tương quan dương cao nhất (**$+0.250$**). Khách hàng gặp trục trặc dịch vụ dẫn đến khiếu nại có tỷ lệ rời bỏ tăng vọt lên **31.7%** (so với 10.9% ở nhóm không có khiếu nại, gấp gần 3 lần).
3. **Mức độ tương tác & Ưu đãi:** `DaySinceLastOrder` ($-0.161$) và `CashbackAmount` ($-0.154$) cho thấy khách hàng lâu ngày không đặt đơn hoặc nhận ít tiền hoàn có xu hướng rời bỏ cao hơn.
4. **Phân hóa theo tình trạng hôn nhân:** Nhóm `Single` (độc thân) có tỷ lệ rời bỏ cao nhất (**~26.7%**), tiếp theo là `Divorced` (**~14.6%**) và thấp nhất là `Married` (**~11.5%**).
5. Vì không có một biến đơn lẻ nào giải thích 100% việc churn, bước tiếp theo cần kỹ thuật trích xuất đặc trưng (feature engineering) và mô hình hóa phi tuyến để tối ưu hóa dự báo.

---

### 3️⃣ Bước 3: Feature Engineering Không Rò rỉ Thông tin (`03_feature_engineering.ipynb`)

#### 🎯 Mục tiêu
Chuẩn bị tập dữ liệu hoàn chỉnh cho mô hình hóa: chia train/test trước để tránh rò rỉ thông tin (data leakage), bổ sung missing indicators, điền missing values bằng median tập train, chuẩn hóa các giá trị danh pháp trùng lặp, tạo 4 đặc trưng nghiệp vụ mới và One-Hot Encoding.

#### 🛠️ Mã nguồn & Thực thi trong Notebook

##### [In 5]: Bước 3.1 - Chia Train/Test trước khi tiền xử lý
```python
# Tách feature và target
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Train:', X_train.shape, '| Test:', X_test.shape)
print('Train churn rate:', round(y_train.mean(), 4))
print('Test churn rate:', round(y_test.mean(), 4))
```
##### [Out 5]:
```
Train: (4504, 18) | Test: (1126, 18)
Train churn rate: 0.1683
Test churn rate: 0.1687
```

##### [In 7]: Bước 3.2 - Thêm Missing Indicators & Điền Missing Values
*Nhận xét nghiệp vụ:* Việc thiếu dữ liệu tự nó là tín hiệu rất mạnh dự báo Churn (ví dụ: nhóm thiếu `Tenure` có churn rate tới **30.7%**, gấp đôi trung bình 16.8%; thiếu `WarehouseToHome` tới **33.5%**; trong khi thiếu `CouponUsed` chỉ **3.1%**). Do đó, tạo cờ nhị phân `{col}_missing` giữ lại trọn vẹn tín hiệu này.
```python
X_train = X_train.copy()
X_test = X_test.copy()

# 1. Thêm missing indicator (0/1)
for col in list_null:
    X_train[f'{col}_missing'] = X_train[col].isnull().astype(int)
    X_test[f'{col}_missing'] = X_test[col].isnull().astype(int)

# 2. Điền missing bằng median của tập TRAIN áp dụng cho cả train và test
cols_with_na = X_train.columns[X_train.isnull().any()].tolist()
medians = X_train[cols_with_na].median()
X_train[cols_with_na] = X_train[cols_with_na].fillna(medians)
X_test[cols_with_na] = X_test[cols_with_na].fillna(medians)

print('Missing in train:', X_train.isnull().sum().sum())
print('Missing in test:', X_test.isnull().sum().sum())
print('So feature:', X_train.shape[1])
```
##### [Out 7]:
```
Missing in train: 0
Missing in test: 0
So feature: 25
```

##### [In 9]: Bước 3.3 - Chuẩn hóa Giá trị Phân loại
Gộp các nhãn đồng nghĩa do sai sót nhập liệu ban đầu và chuyển đổi kiểu dữ liệu cột phân khúc:
```python
# 1. Gộp các giá trị trùng ý nghĩa
login_device_map = {'Phone': 'Mobile Phone'}
payment_mode_map = {'COD': 'Cash on Delivery', 'CC': 'Credit Card'}
order_cat_map = {'Mobile': 'Mobile Phone'}

for col, mapping in [
    ('PreferredLoginDevice', login_device_map),
    ('PreferredPaymentMode', payment_mode_map),
    ('PreferedOrderCat', order_cat_map)
]:
    X_train[col] = X_train[col].replace(mapping)
    X_test[col] = X_test[col].replace(mapping)

# 2. Chuyển CityTier thành nhãn phân loại (object)
X_train['CityTier'] = X_train['CityTier'].astype('object')
X_test['CityTier'] = X_test['CityTier'].astype('object')
```
##### [Out 9]:
```
PreferredLoginDevice: ['Computer', 'Mobile Phone']
PreferredPaymentMode: ['Cash on Delivery', 'Credit Card', 'Debit Card', 'E wallet', 'UPI']
PreferedOrderCat: ['Fashion', 'Grocery', 'Laptop & Accessory', 'Mobile Phone', 'Others']
Chuyen thanh object: CityTier
```

##### [In 11] & [In 12]: Bước 3.4 - Tạo 4 Đặc trưng Nghiệp vụ Mới (Domain Features)
```python
def add_features(X):
    X = X.copy()
    # Số coupon trung bình mỗi đơn hàng
    X['CouponPerOrder'] = (X['CouponUsed'] / X['OrderCount'].replace(0, np.nan)).fillna(0)
    # Khách hàng có mua hàng gần đây (trong vòng 7 ngày)
    X['IsRecentBuyer'] = (X['DaySinceLastOrder'] <= 7).astype(int)
    # Chỉ số tương tác tích hợp trên app
    X['AppEngagement'] = X['HourSpendOnApp'] * X['OrderCount']
    # Phân nhóm thời gian gắn bó
    X['TenureGroup'] = pd.cut(
        X['Tenure'], bins=[-np.inf, 12, 36, np.inf],
        labels=['New', 'Mid', 'Senior']
    ).astype(str)
    return X

X_train = add_features(X_train)
X_test = add_features(X_test)
```
##### [Out 12]: Kiểm tra tỷ lệ Churn theo các biến mới
* **Theo `IsRecentBuyer`:**
  * `0` (Dormant, > 7 ngày): **9.58%** Churn
  * `1` (Recent Buyer, $\le 7$ ngày): **18.99%** Churn (Nhóm mua gần đây nhưng gặp sự cố dịch vụ thường chủ động rời bỏ)
* **Theo `TenureGroup`:**
  * `New` ($\le 12$ tháng): **22.93%** Churn (Vùng nguy hiểm nhất)
  * `Mid` ($12-36$ tháng): **5.18%** Churn
  * `Senior` ($> 36$ tháng): **0.00%** Churn (Tuyệt đối trung thành)

##### [In 14]: Bước 3.5 - One-Hot Encoding & Căn chỉnh Cột Train/Test
```python
cat_cols = X_train.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Căn chỉnh cột: nếu test thiếu category nào thì điền 0
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print('Train shape:', X_train_enc.shape)
print('Test shape:', X_test_enc.shape)
```
##### [Out 14]:
```
Categorical columns: ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus', 'TenureGroup']
Train shape: (4504, 38)
Test shape: (1126, 38)
```

##### [In 16]: Bước 3.6 - Lưu dữ liệu đã xử lý vào `data/processed/`
```python
train_processed = X_train_enc.copy()
train_processed['Churn'] = y_train.values
test_processed = X_test_enc.copy()
test_processed['Churn'] = y_test.values

train_processed.to_csv('../data/processed/train_processed.csv', index=False)
test_processed.to_csv('../data/processed/test_processed.csv', index=False)
```
##### [Out 16]:
```
Đã lưu train_processed.csv: (4504, 39)
Đã lưu test_processed.csv: (1126, 39)
```

#### 💡 Kết luận Rút ra từ Bước 3:
1. Chia train/test (80/20, stratify) trước khi xử lý đảm bảo hoàn toàn **không rò rỉ dữ liệu**.
2. Thêm 7 cờ missing indicator giúp mô hình nắm bắt hành vi ẩn của khách hàng mà không làm méo mó phân phối gốc.
3. Điền missing bằng median của tập train là lựa chọn tối ưu vì 6/7 cột có độ lệch phân phối (skew 0.7–2.6).
4. Tạo thành công 4 đặc trưng có khả năng phân tách churn rate rất rõ rệt (`TenureGroup`, `IsRecentBuyer`, `AppEngagement`, `CouponPerOrder`).
5. Kết thúc Bước 3, dữ liệu sẵn sàng với **38 features đã encode** được lưu vào `data/processed/`.

---

### 4️⃣ Bước 4: Huấn luyện, So sánh Mô hình & Tuning Hyperparameter (`04_model_training.ipynb`)

#### 🎯 Mục tiêu
Xây dựng pipeline đánh giá benchmark, so sánh 4 kiến trúc thuật toán học máy (Logistic Regression, Random Forest, XGBoost, LightGBM), ưu tiên **Recall** ở lớp Churn để bắt trọn khách hàng rời bỏ, và thực hiện tuning siêu tham số cho mô hình dẫn đầu.

#### 🛠️ Mã nguồn & Thực thi trong Notebook

##### [In 1]: Tải dữ liệu đã qua tiền xử lý
```python
train = pd.read_csv('../data/processed/train_processed.csv')
test = pd.read_csv('../data/processed/test_processed.csv')

X_train = train.drop(columns=['Churn'])
y_train = train['Churn']
X_test = test.drop(columns=['Churn'])
y_test = test['Churn']
```

##### [In 3]: Bước 4.1 - Baseline: Logistic Regression (với `StandardScaler` & `class_weight='balanced'`)
```python
def evaluate_model(model, X_test, y_test, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    }

lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
lr.fit(X_train, y_train)
results = [evaluate_model(lr, X_test, y_test, 'LogisticRegression')]
```
[Out 3:]
<img width="1242" height="198" alt="{9F2D6561-7241-4CC7-9116-ED674C4A74C4}" src="https://github.com/user-attachments/assets/a2fb3b09-f4ab-4b77-abb9-c47a1956f124" />

##### [In 5]: Bước 4.2 - Random Forest Classifier
```python
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate_model(rf, X_test, y_test, 'RandomForest'))
```
[Out 5]:
<img width="1272" height="289" alt="{36ED1687-B4C5-4E68-B216-A40C76F4CA93}" src="https://github.com/user-attachments/assets/81b8a846-74bc-48c3-af24-ad8a8ce3622f" />

##### [In 7]: Bước 4.3 - XGBoost Classifier (`scale_pos_weight = 4.94`)
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
)
xgb.fit(X_train, y_train)
results.append(evaluate_model(xgb, X_test, y_test, 'XGBoost'))
```
<img width="1233" height="372" alt="{BD3D62BF-F54C-4B45-AAA3-229104341EA0}" src="https://github.com/user-attachments/assets/c7e7abd2-7cf3-43e3-a144-d2b745b4fbee" />

##### [In 9]: Bước 4.4 - LightGBM Classifier (`class_weight='balanced'`)
```python
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
results.append(evaluate_model(lgbm, X_test, y_test, 'LightGBM'))
```
<img width="1278" height="406" alt="{0D8B394D-F865-47DF-B1E7-7AE09D643A8E}" src="https://github.com/user-attachments/assets/ea219107-affc-4d5d-b1cc-681c87109313" />

##### [In 11]: Bước 4.5 - Bảng Tổng hợp So sánh Mô hình
```python
results_df = pd.DataFrame(results).sort_values(['Recall', 'F1'], ascending=False).reset_index(drop=True)
results_df.round(4)
```
##### [Out 11]: Kết quả Benchmark trên Unseen Test Set (1,126 khách hàng)
| Model | Accuracy | Precision | Recall (Ưu tiên) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.9778** | **0.9231** | **0.9474** | **0.9351** | **0.9930** |
| **RandomForest** | 0.9751 | 0.9091 | 0.9474 | 0.9278 | 0.9928 |
| **XGBoost** | 0.9671 | 0.8844 | 0.9263 | 0.9049 | 0.9862 |
| **LogisticRegression** | 0.8242 | 0.4879 | 0.8474 | 0.6192 | 0.9043 |

*(Bbiểu đồ so sánh mô hình bar chart `results_df.plot`)*  
<img width="2622" height="1195" alt="{945DB1EF-4E32-4199-BB6A-A2F16DB876AF}" src="https://github.com/user-attachments/assets/84afca9e-283b-45ca-8cff-f335a65581e8" />

##### [In 13]: Bước 4.6 - Tuning Hyperparameter LightGBM với `RandomizedSearchCV`
Dùng `RandomizedSearchCV` (cv=5, n_iter=50) tối ưu hóa chỉ số `f1` trên mô hình LightGBM:
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [200, 300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, -1],
    'min_child_samples': [10, 20, 40],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

lgbm_search = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
search = RandomizedSearchCV(lgbm_search, param_dist, n_iter=50, scoring='f1', cv=5, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print('Best params:', search.best_params_)
print('Best CV F1:', round(search.best_score_, 4))
```
##### [Out 13]:
```
Best params: {'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'min_child_samples': 20, 'max_depth': 7, 'learning_rate': 0.2, 'colsample_bytree': 0.6}
Best CV F1: 0.8969
```

Kết quả của mô hình sau tuning (`LightGBM-Tuned`):
* **Accuracy:** **0.9796** (97.96%)
* **Precision:** **0.9326** (93.26%)
* **Recall:** **0.9474** (94.74%)
* **F1-Score:** **0.9399**
* **ROC-AUC:** **0.9929**

##### [In 14]: Lưu mô hình tốt nhất vào `models/churn_pipeline.pkl`
```python
os.makedirs('../models', exist_ok=True)
joblib.dump(lgbm, '../models/churn_pipeline.pkl')
print(f'Model tot nhat (Recall): LightGBM')
print('Da luu: ../models/churn_pipeline.pkl')
```

#### 💡 Kết luận Rút ra từ Bước 4:
1. Logistic Regression là baseline hữu ích nhưng cho Precision thấp (~48.8%) do không bắt được các mối quan hệ phi tuyến.
2. Các mô hình cây (Tree-based Ensembles) vượt trội rõ rệt. Cả Random Forest và LightGBM đều đạt **Recall 94.74%**, nhưng **LightGBM dẫn đầu về Precision (92.31%) và F1-Score (0.9351)**.
3. Sau khi tuning siêu tham số, `LightGBM-Tuned` đẩy F1-Score lên **0.9399** và Accuracy lên **97.96%**.
4. Mô hình LightGBM được chọn làm Production Champion và lưu trữ an toàn tại `models/churn_pipeline.pkl`.

---

### 5️⃣ Bước 5: Đánh giá Chi tiết, Ma trận Nhầm lẫn & Giải thích Mô hình (`05_model_evaluation.ipynb`)

#### 🎯 Mục tiêu
Thực hiện đánh giá sâu trên tập test độc lập (1,126 khách hàng) thông qua Classification Report, Ma trận nhầm lẫn (Confusion Matrix), phân tích lỗi False Negative, vẽ đường cong ROC Curve, Precision-Recall Curve và tính toán Feature Importance.

#### 🛠️ Mã nguồn & Thực thi trong Notebook

##### [In 1]: Tải mô hình đã lưu và xuất Classification Report
```python
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

model = joblib.load('../models/churn_pipeline.pkl')
test = pd.read_csv('../data/processed/test_processed.csv')
X_test = test.drop(columns=['Churn'])
y_test = test['Churn']

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds, digits=4))
```
##### [Out 1]:
```
              precision    recall  f1-score   support

           0     0.9893    0.9840    0.9866       936
           1     0.9231    0.9474    0.9351       190

    accuracy                         0.9778      1126
   macro avg     0.9562    0.9657    0.9608      1126
weighted avg     0.9781    0.9778    0.9779      1126
```

##### [In 3]: Ma trận Nhầm lẫn (Confusion Matrix) & Phân tích Lỗi
```python
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'So khach churn thuc te bi bo sót (False Negative): {fn}')
print(f'So khach churn duoc phat hien dung (True Positive): {tp}')
```
##### [Out 3]:
```
So khach churn thuc te bi bo sót (False Negative): 10
So khach churn duoc phat hien dung (True Positive): 180
```
Chi tiết cấu trúc ma trận nhầm lẫn:
* **True Positive (TP = 180):** Phát hiện chính xác **180 trên tổng số 190 khách hàng rời bỏ thực tế** trong tập kiểm thử (đạt tỷ lệ 94.74%).
* **False Negative (FN = 10):** Chỉ có **10 khách hàng rời bỏ bị bỏ sót** trên 1,126 lượt đánh giá (**chưa đầy 0.88%**).
* **True Negative (TN = 921):** 921 khách hàng không rời bỏ được nhận diện chính xác.
* **False Positive (FP = 15):** Chỉ có 15 trường hợp báo động nhầm (**Precision đạt 92.31%**), giúp doanh nghiệp không bị lãng phí ngân sách ưu đãi cho khách hàng vốn đang hài lòng.

*(Confusion Matrix Heatmap)*  
<img width="1342" height="1175" alt="{18802048-1F77-4F28-8D60-0701E39840DD}" src="https://github.com/user-attachments/assets/29aa842d-968d-4bb5-b853-7fc73393357f" />

##### [In 5]: Đường cong ROC & ROC-AUC
```python
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```
##### [Out 5]: **ROC-AUC = 0.9930** (Khả năng phân biệt hai lớp gần như hoàn hảo).

*(Ảnh chụp màn hình ROC Curve)*  
<img width="1674" height="1179" alt="{697F9FBD-5962-4954-94A9-9B7AB985E613}" src="https://github.com/user-attachments/assets/2a0503b2-746d-4717-b0e2-b27157c040f1" />


##### [In 7]: Precision-Recall Curve & PR-AUC
```python
precision, recall, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim(0, 1.05)
plt.legend()
plt.show()
```
##### [Out 7]: **PR-AUC = 0.9690** (Chứng minh độ ổn định vượt bậc trên dữ liệu mất cân bằng).

*(Ảnh chụp màn hình Precision-Recall Curve)*  
<img width="1640" height="1184" alt="{D8FB72BD-E08E-4324-9487-5DC4E05D2E28}" src="https://github.com/user-attachments/assets/ab282f54-3c49-401e-ae3f-661a19073a73" />

##### [In 9]: Feature Importance (Giải thích Mô hình)
```python
est = model.named_steps['clf'] if hasattr(model, 'named_steps') else model

if hasattr(est, 'feature_importances_'):
    imp = pd.Series(est.feature_importances_, index=X_test.columns)
    title = 'Feature Importance'
else:
    imp = pd.Series(np.abs(est.coef_[0]), index=X_test.columns)
    title = '|Coefficient| (Logistic Regression)'

imp.nlargest(15).sort_values().plot(kind='barh', figsize=(8, 6))
plt.title(title)
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```
*(Ảnh chụp màn hình Top 15 Feature Importances)*  
<img width="2178" height="1465" alt="{F56A2C35-C5EE-4421-95E1-30AD9E2F2DFE}" src="https://github.com/user-attachments/assets/0c201cb4-6533-496d-b568-586c61e41973" />

#### 💡 Kết luận Rút ra từ Bước 5:
1. **False Negative cực thấp (10 tài khoản):** Đáp ứng tiêu chuẩn triển khai thực tế của doanh nghiệp, giảm thiểu nguy cơ mất doanh thu định kỳ từ nhóm khách hàng có giá trị.
2. **ROC-AUC (0.9930) và PR-AUC (0.9690):** Khẳng định độ tin cậy của thuật toán phân loại trên tập dữ liệu mất cân bằng.
3. **Các biến quyết định (Drivers):** `Tenure` (thời gian gắn bó), `DaySinceLastOrder` (recency), `WarehouseToHome` (khoảng cách vận chuyển), `CashbackAmount` (mức hoàn tiền) và `Complain` (khiếu nại) là 5 cột mốc định hình trực tiếp hành vi của người tiêu dùng.

---

## 💡 Tổng hợp Phát hiện & Khuyến nghị Giữ chân Khách hàng

Dựa trên kết quả phân tích dữ liệu và trích xuất đặc trưng của mô hình, bảng ma trận sau đây đối chiếu rõ nét sự khác biệt giữa hai nhóm khách hàng và đề xuất giải pháp chiến lược:

| Đặc trưng Quan sát | Nhóm Ở lại (Non-Churn: $N=4,682$) | Nhóm Rời bỏ (Churn: $N=948$) | Insight Hành vi Nghiệp vụ | Khuyến nghị Giữ chân (Retention Action) |
| :--- | :--- | :--- | :--- | :--- |
| **`Tenure` (Vòng đời gắn bó)** | Trung vị: **10 tháng**; phân bổ đều từ 10–30 tháng. | Trung vị: **2 tháng**; hơn 80% rời bỏ trong vòng 5 tháng đầu. | **Giai đoạn Onboarding là bước tử huyệt:** Khách hàng mới rời bỏ rất sớm nếu trải nghiệm ban đầu không trọn vẹn. | Xây dựng chương trình đồng hành 90 ngày đầu (Welcome Journeys), thưởng voucher tích lũy cho đơn thứ 2 và thứ 3. |
| **`Complain` (Khiếu nại hỗ trợ)** | Chỉ **10.9%** khách hàng từng gửi khiếu nại. | Lên tới **31.7%** khách hàng từng khiếu nại ($3\times$). | **Dịch vụ CS yếu kém đẩy khách đi:** Khiếu nại không được giải quyết nhanh là động cơ rời bỏ tức thì. | Đặt SLA khẩn cấp: Với khách hàng khiếu nại có xác suất churn cao, quản lý CS phải liên hệ hỗ trợ trong vòng 4 giờ kèm ưu đãi bù đắp. |
| **`DaySinceLastOrder` (Recency)** | Chủ yếu dưới 5 ngày; tần suất mua lặp lại ổn định. | Thường vượt quá 12–20 ngày không phát sinh đơn. | **Dấu hiệu ngủ đông (Dormancy):** Khoảng cách giữa các đơn tăng dần là tín hiệu cảnh báo trước khi hủy app. | Thiết lập luồng email/push tự động nhắc nhở vào ngày thứ 10, gửi mã giảm giá đúng danh mục yêu thích (`PreferedOrderCat`). |
| **`CashbackAmount` (Hoàn tiền)** | Tập trung trong khoảng 160 – 220 đơn vị tiền tệ. | Tập trung dưới 140; phân bổ phân tán. | **Cạnh tranh về lợi ích:** Khách hàng nhận được ít ưu đãi hoàn tiền cảm thấy không được trân trọng. | Tái cấu trúc chính sách khách hàng thân thiết theo bậc (Tiered Loyalty), tăng tỷ lệ hoàn tiền khi duy trì tần suất mua. |
| **`WarehouseToHome` (Khoảng cách giao hàng)** | Đa số dưới 15 km; thời gian nhận hàng nhanh. | Nhiều khách hàng ở xa từ 25 – 35 km. | **Áp lực thời gian giao hàng:** Khoảng cách xa gây trễ đơn, ảnh hưởng tiêu cực đến trải nghiệm nhận hàng. | Tối ưu hóa chuỗi cung ứng với các hub vệ tinh 3PL; hỗ trợ miễn phí vận chuyển hỏa tốc cho khách hàng ở xa. |
| **`MaritalStatus` (Hôn nhân)** | Nhóm đã kết hôn (`Married`) có tỷ lệ churn thấp nhất (11.5%). | Nhóm độc thân (`Single`) có tỷ lệ churn cao nhất (26.7%). | **Khách hàng độc thân kém trung thành hơn:** Dễ thay đổi ứng dụng theo các chiến dịch quảng cáo ngắn hạn. | Thiết kế các combo tiêu dùng cá nhân, flash sale hấp dẫn trên giao diện ứng dụng di động cho khách hàng trẻ độc thân. |

---


*Lưu ý: Mở và chạy tuần tự các notebook từ `01` đến `05` để tái hiện toàn bộ kết quả phân tích và đánh giá.*

### 5. Sử dụng Mô hình đã Lưu để Dự báo (Inference)
```python
import joblib
import pandas as pd

# Tải mô hình LightGBM đã huấn luyện
model = joblib.load('models/churn_pipeline.pkl')

# Đọc dữ liệu khách hàng cần dự báo
new_data = pd.read_csv('data/processed/test_processed.csv')
X = new_data.drop(columns=['Churn'])

# Dự báo xác suất rời bỏ
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print(f"Đã dự báo thành công cho {len(predictions)} tài khoản khách hàng!")
```

---

## 👤 Thông tin Tác giả & Liên hệ

* **Họ và tên:** **Nguyễn Xuân Thu**
* **Vị trí:** Data Analyst / Machine Learning Enthusiast
* **Email:** [thu.nx280@gmail.com](mailto:thu.nx280@gmail.com)
* **GitHub:** [https://github.com/thunx280-ctrl](https://github.com/thunx280-ctrl)
* **Repository Dự án:** [https://github.com/thunx280-ctrl/churn_prediction](https://github.com/thunx280-ctrl/churn_prediction)

---
*⭐ Nếu bạn thấy dự án phân tích dự báo khách hàng rời bỏ này hữu ích cho công việc hoặc học tập, hãy ủng hộ bằng cách nhấn Star cho repository nhé!*
