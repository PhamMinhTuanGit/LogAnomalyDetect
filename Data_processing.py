import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

# --- CẤU HÌNH ---
# Vui lòng thay đổi các giá trị này cho phù hợp với dữ liệu của bạn
DATA_DIR = 'data'                             # Thư mục chứa các file Excel
TIMESTAMP_COLUMN = 'timestamp'                # Tên cột thời gian, đặt là None nếu không có
OUTPUT_FILE = 'processed_data.npy'            # Tên file output

WINDOW_SIZE = 30
OVERLAP = 29
STEP = WINDOW_SIZE - OVERLAP # = 1

def create_mock_data_if_needed(directory):
    """Tạo dữ liệu giả nếu thư mục data không có file excel nào."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    excel_files = glob.glob(os.path.join(directory, '*.xlsx'))
    if not excel_files:
        print(f"Cảnh báo: Không tìm thấy file Excel trong '{directory}'. Đang tạo dữ liệu giả...")
        mock_file_path = os.path.join(directory, 'mock_timeseries_data.xlsx')
        num_rows = 200
        num_features = 4
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='h'))
        
        data = {TIMESTAMP_COLUMN: dates}
        for i in range(num_features):
            # Tạo ra một vài dao động và nhiễu
            seasonal_component = np.sin(np.linspace(0, 8 * np.pi, num_rows)) * (i + 1) * 5
            noise = np.random.randn(num_rows) * 2
            data[f'feature_{i+1}'] = 50 + seasonal_component + noise + i * 10
        
        mock_df = pd.DataFrame(data)
        mock_df.to_excel(mock_file_path, index=False)
        print(f"Đã tạo file dữ liệu giả tại: '{mock_file_path}'")

def load_all_data_from_directory(directory):
    """Đọc và gộp tất cả các file Excel từ một thư mục."""
    search_path = os.path.join(directory, '*.xlsx')
    excel_files = glob.glob(search_path)
    
    if not excel_files:
        return pd.DataFrame() # Trả về DataFrame rỗng nếu không có file

    df_list = []
    print(f"Tìm thấy {len(excel_files)} file Excel. Đang tiến hành đọc...")
    for file in excel_files:
        try:
            df_list.append(pd.read_excel(file))
        except Exception as e:
            print(f"Lỗi khi đọc file {file}: {e}")
    
    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

# --- BẮT ĐẦU XỬ LÝ ---
create_mock_data_if_needed(DATA_DIR)

# 1. Đọc tất cả các file Excel trong thư mục và gộp lại
df = load_all_data_from_directory(DATA_DIR)

if df.empty:
    print(f"Lỗi: Không có dữ liệu để xử lý trong thư mục '{DATA_DIR}'.")
    exit()

# 2. Xử lý và sắp xếp dữ liệu theo thời gian
if TIMESTAMP_COLUMN and TIMESTAMP_COLUMN in df.columns:
    print(f"Xử lý cột thời gian '{TIMESTAMP_COLUMN}'...")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df = df.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
    # Tách các cột dữ liệu số
    numeric_df = df.drop(columns=[TIMESTAMP_COLUMN])
else:
    print("Không có cột thời gian được chỉ định hoặc không tìm thấy. Chỉ xử lý các cột số.")
    # Nếu không có cột thời gian, chỉ chọn các cột số
    numeric_df = df.select_dtypes(include=np.number)

# Kiểm tra xem có cột số nào không
if numeric_df.empty:
    print("Lỗi: Không tìm thấy cột dữ liệu dạng số nào để xử lý.")
    exit()

print(f"Các cột dữ liệu được xử lý: {list(numeric_df.columns)}")

# 3. Chuẩn hóa các cột dữ liệu dạng số
print("Chuẩn hóa dữ liệu bằng StandardScaler...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# 4. Tạo sliding window
print(f"Tạo sliding window (size={WINDOW_SIZE}, overlap={OVERLAP})...")
windows = []
n_samples = len(scaled_data)

# Vòng lặp để tạo các cửa sổ
for i in range(0, n_samples - WINDOW_SIZE + 1, STEP):
    window = scaled_data[i : i + WINDOW_SIZE]
    windows.append(window)

if not windows:
    print(f"Lỗi: Dữ liệu quá ngắn ({n_samples} điểm) để tạo cửa sổ với kích thước {WINDOW_SIZE}.")
    exit()

# 5. Chuyển đổi sang mảng NumPy
processed_array = np.array(windows)

print(f"Hoàn tất. Shape của mảng kết quả: {processed_array.shape}")
# Shape sẽ là (số window, 30, số biến)

# 6. Lưu kết quả vào file .npy
np.save(OUTPUT_FILE, processed_array)
print(f"Đã lưu kết quả vào file '{OUTPUT_FILE}'")

# Kiểm tra lại file đã lưu
loaded_data = np.load(OUTPUT_FILE)
print(f"Kiểm tra file đã lưu: shape là {loaded_data.shape}, kiểu dữ liệu là {loaded_data.dtype}")
