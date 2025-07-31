import re
import pandas as pd
import numpy as np
import torch
import os
import glob

def convert_value(value):
    """
    Chuyển đổi một giá trị đơn lẻ thành float, xử lý chuỗi, số và các giá trị rỗng.
    """
    if pd.isna(value):
        return -1.0
    if isinstance(value, (int, float)):
        return float(value)
    
    s_value = str(value).strip()
    if s_value.lower() in ["none", "n/a", "--", ""]:
        return -1.0
    
    # Bỏ các ký tự không phải số (giữ lại dấu chấm và dấu trừ ở đầu)
    s_value = re.sub(r"[,%a-zA-Z_]", "", s_value)
    try:
        return float(s_value)
    except (ValueError, TypeError):
        return -1.0

def extract_percentage(value):
    """
    Trích xuất giá trị phần trăm từ chuỗi có dạng '...(...%)'.
    Ví dụ: '220MB(29%)' -> 29.0
    """
    if not isinstance(value, str):
        return -1.0

    # Tìm kiếm số bên trong dấu ngoặc đơn và theo sau là ký tự %
    match = re.search(r'\((\d+(\.\d+)?)\s*%\)', value)
    if match:
        try:
            # group(1) sẽ là toàn bộ số (ví dụ: '29' hoặc '29.5')
            return float(match.group(1))
        except (ValueError, TypeError):
            return -1.0
    return -1.0

def parse_int_info(text_block):
    """
    Phân tích một khối văn bản từ cột 'int_info' và trích xuất các giá trị số.
    Trả về một dictionary chứa các giá trị đã trích xuất.
    """
    if not isinstance(text_block, str):
        return {}

    fields_to_extract = ["port_temp", "voltage", "rx_power", "tx_power", "rxbs", "int_ut", "txbs", "out_ut"]
    # Khởi tạo kết quả với giá trị mặc định
    extracted_values = {field: -1.0 for field in fields_to_extract}

    for line in text_block.strip().split('\n'):
        for field in fields_to_extract:
            # Tìm kiếm linh hoạt hơn, không phân biệt chữ hoa/thường
            match = re.search(fr"{field}:\s*([^,]+)", line, re.IGNORECASE)
            if match:
                # Sử dụng lại hàm convert_value để chuẩn hóa giá trị
                value_str = match.group(1)
                extracted_values[field] = convert_value(value_str)
    
    return extracted_values

def process_excel_file(file_path, columns_to_drop):
    """
    Hàm chính để đọc, xử lý và chuyển đổi file Excel.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Đọc thành công file: {file_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return None

    # A. Tạo cột 'label' dựa trên 'pkt_loss_rate' trước khi xử lý các cột khác
    if 'pkt_loss_rate' in df.columns:
        # Chuyển đổi cột pkt_loss_rate thành số để so sánh
        numeric_pkt_loss_rate = df['pkt_loss_rate'].apply(convert_value)
        # Tạo cột label: 1 nếu > 1, ngược lại là 0
        df['label'] = (numeric_pkt_loss_rate > 1).astype(int)
    else:
        # Nếu không có cột pkt_loss_rate, mặc định tất cả là bình thường (0)
        df['label'] = 0

    # 1. Xử lý cột 'int_info'
    int_info_series = df['int_info'].apply(parse_int_info)
    # Chuyển đổi Series của dictionaries thành DataFrame
    int_info_df = pd.DataFrame(int_info_series.tolist())

    # 2. Xử lý các cột còn lại
    # Bỏ các cột không cần thiết và cột 'int_info' đã xử lý
    other_info_df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Các cột đặc biệt cần trích xuất phần trăm
    percentage_columns = ['memused', 'memfree']

    # Áp dụng hàm convert_value cho tất cả các cột còn lại
    for col in other_info_df.columns:
        if col in ['timestamp', 'label']: # Giữ nguyên cột timestamp và label, không chuyển đổi
            continue
        if col in percentage_columns:
            other_info_df[col] = other_info_df[col].apply(extract_percentage)
        else:
            other_info_df[col] = other_info_df[col].apply(convert_value)

    # 3. Gộp hai DataFrame đã xử lý
    # Đảm bảo thứ tự cột timestamp được giữ lại nếu có
    processed_df = pd.concat([other_info_df, int_info_df], axis=1)
    
    return processed_df

if __name__ == "__main__":
    # --- CẤU HÌNH ---
    DATA_DIR = 'data'
    OUTPUT_TENSOR_FILE = 'data.pt'

    # Các cột cần loại bỏ (bao gồm cả cột sẽ được xử lý riêng)
    # Cột 'timestamp' được giữ lại để thực hiện gộp nhóm
    DROP_COLUMNS = [
        "int_info", "fan_1", "fan_2", "fan_3", "vol_1_35V", "vol_1V",
        "vol_1_3V_OCXO", "vol_1_3V_Sys", "vol_12V", "device", "hostname",
        "uptime", "pkt_loss", "pkt_loss_rate"
    ]

    # 1. Đọc và xử lý tất cả các file Excel trong thư mục data
    all_files = glob.glob(os.path.join(DATA_DIR, '*.xlsx'))
    if not all_files:
        print(f"Lỗi: Không tìm thấy file Excel nào trong thư mục '{DATA_DIR}'.")
        exit()

    print(f"Tìm thấy {len(all_files)} file(s). Bắt đầu xử lý...")
    
    # Sử dụng vòng lặp for rõ ràng để xử lý từng file một
    processed_dfs_list = []
    for file_path in all_files:
        print(f"\n--- Đang xử lý file: {os.path.basename(file_path)} ---")
        # Gọi hàm xử lý cho một file duy nhất
        single_processed_df = process_excel_file(file_path, DROP_COLUMNS)
        
        # Chỉ thêm vào danh sách nếu xử lý thành công
        if single_processed_df is not None:
            processed_dfs_list.append(single_processed_df)

    if not processed_dfs_list:
        print("Không có dữ liệu nào được xử lý thành công.")
        exit()

    # 2. Gộp tất cả các DataFrame đã xử lý thành một
    print("\n--- Gộp dữ liệu từ tất cả các file đã xử lý ---")
    combined_df = pd.concat(processed_dfs_list, ignore_index=True)

    # 3. Gộp các hàng có cùng timestamp
    print("\n--- Gộp các hàng có cùng timestamp trên tất cả các file ---")
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    combined_df.dropna(subset=['timestamp'], inplace=True) # Bỏ các hàng có timestamp không hợp lệ

    # Xác định cách gộp cho từng cột: sum cho features, max cho label
    feature_cols = [col for col in combined_df.columns if col not in ['timestamp', 'label']]
    agg_dict = {col: 'sum' for col in feature_cols}
    agg_dict['label'] = 'max'  # Nếu có bất kỳ hàng nào là 1 (bất thường), kết quả sẽ là 1

    aggregated_df = combined_df.groupby('timestamp').agg(agg_dict).reset_index()
    print(f"Tổng số hàng (timestamp duy nhất) sau khi gộp: {len(aggregated_df)}")

    # 4. Chuyển đổi sang Tensor
    print("\n--- Chuyển đổi sang Tensor PyTorch ---")
    # Tách features và labels ra khỏi DataFrame đã gộp
    labels_df = aggregated_df['label']
    features_df = aggregated_df.drop(columns=['timestamp', 'label'])

    # Chuyển đổi sang tensor
    features_tensor = torch.from_numpy(features_df.to_numpy()).float()
    labels_tensor = torch.from_numpy(labels_df.to_numpy()).long() # Labels thường là kiểu Long

    print("Đã chuyển đổi thành công!")
    print(f"Shape của Tensor Features: {features_tensor.shape}")
    print(f"Shape của Tensor Labels: {labels_tensor.shape}")

    # 5. Lưu tensor kết quả ra file
    # Lưu dưới dạng một dictionary để dễ dàng truy cập features và labels
    torch.save({'features': features_tensor, 'labels': labels_tensor}, OUTPUT_TENSOR_FILE)
    print(f"\nĐã lưu features và labels vào file: '{OUTPUT_TENSOR_FILE}'")