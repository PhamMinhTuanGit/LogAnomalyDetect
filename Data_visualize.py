import numpy as np
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH ---
INPUT_FILE = 'processed_data.npy' # File dữ liệu đã được xử lý từ Data_processing.py
NUM_WINDOWS_TO_PLOT = 3           # Số lượng window mẫu cần vẽ

def visualize_data():
    """
    Tải và trực quan hóa dữ liệu chuỗi thời gian từ file .npy.
    """
    # 1. Kiểm tra sự tồn tại của file dữ liệu
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: File '{INPUT_FILE}' không tồn tại.")
        print("Vui lòng chạy file 'Data_processing.py' trước để tạo dữ liệu.")
        return

    print(f"Đang tải dữ liệu từ '{INPUT_FILE}'...")
    try:
        data = np.load(INPUT_FILE)
    except Exception as e:
        print(f"Lỗi khi tải file: {e}")
        return

    # 2. Lấy thông tin về dữ liệu
    num_windows, window_size, num_features = data.shape
    print(f"Dữ liệu có shape: (số window={num_windows}, kích thước window={window_size}, số biến={num_features})")

    if num_windows == 0:
        print("File dữ liệu rỗng, không có gì để vẽ.")
        return

    # 3. Vẽ biểu đồ cho một vài window mẫu
    indices_to_plot = range(min(NUM_WINDOWS_TO_PLOT, num_windows))

    for i in indices_to_plot:
        window_data = data[i] # Lấy dữ liệu của một window, shape: (window_size, num_features)
        plt.figure(figsize=(12, 6))
        
        for feature_idx in range(num_features):
            plt.plot(window_data[:, feature_idx], label=f'Feature {feature_idx + 1}')

        plt.title(f'Visualization of Window {i}')
        plt.xlabel('Time Step within Window')
        plt.ylabel('Standardized Value')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    visualize_data()
