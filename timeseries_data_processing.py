import re
import pandas as pd
import numpy as np
import torch
import os
import glob

# --- CẤU HÌNH ---
DATA_DIR = 'data'
TARGET_HOSTNAME = 'DUT11HW'
OUTPUT_FILE = 'dut11hw_processed_data.pt'
# Xác định các cổng và trường dữ liệu cần trích xuất
PORTS_OF_INTEREST = ['xe1/1', 'ge1/2']
FIELDS_OF_INTEREST = ['rxbs', 'txbs']

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
    
    # Bỏ các ký tự không phải số (ví dụ: 'bytes/sec', '%', ',', '()')
    s_value = re.sub(r"[a-zA-Z/,%()]", "", s_value)
    try:
        return float(s_value)
    except (ValueError, TypeError):
        return -1.0

def parse_dut11hw_int_info(text_block):
    """
    Phân tích khối văn bản 'int_info' để trích xuất rxbs và txbs cho các cổng cụ thể.
    """
    if not isinstance(text_block, str):
        return [] # Trả về list rỗng cho đầu vào không hợp lệ

    # Dictionary để lưu kết quả tìm thấy, ví dụ: {'xe1/1_rxbs': 123.0}
    extracted_data = {}
    lines = text_block.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Tách tên cổng (ví dụ: 'xe1/1') ra khỏi phần còn lại của dòng
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
        
        port_name = parts[0].strip()
        data_part = parts[1].strip()

        if port_name in PORTS_OF_INTEREST:
            # Nếu tìm thấy port quan tâm, trích xuất các trường dữ liệu từ phần data
            for field in FIELDS_OF_INTEREST:
                # Regex tìm 'field: value' và lấy 'value' cho đến khi gặp field tiếp theo hoặc cuối dòng
                match = re.search(fr'{field}:\s*(.*?)(?=\s*,\s*\w+:|$)', data_part, re.IGNORECASE)
                if match:
                    value_str = match.group(1).strip()
                    numeric_value = convert_value(value_str)
                    extracted_data[f'{port_name}_{field}'] = numeric_value

    # Sắp xếp kết quả theo thứ tự mong muốn: [rxbs_xe1/1, rxbs_ge1/1, txbs_xe1/1, txbs_ge1/1]
    feature_vector = []
    for field in FIELDS_OF_INTEREST:  # Đầu tiên là 'rxbs', sau đó là 'txbs'
        for port in PORTS_OF_INTEREST:  # Đầu tiên là 'xe1/1', sau đó là 'ge1/1'
            key = f'{port}_{field}'
            feature_vector.append(extracted_data.get(key, -1.0))
    
    # Chỉ trả về vector nếu tìm thấy ít nhất một giá trị hợp lệ (khác -1.0)
    if all(v == -1.0 for v in feature_vector):
        return []

    return feature_vector

if __name__ == "__main__":
    print("Bắt đầu quá trình xử lý dữ liệu cho LSTM Autoencoder...")
    all_files = glob.glob(os.path.join(DATA_DIR, '*.xlsx'))
    if not all_files:
        print(f"Lỗi: Không tìm thấy file Excel nào trong thư mục '{DATA_DIR}'.")
        exit()
    # Sắp xếp danh sách file để đảm bảo thứ tự xử lý nhất quán
    all_files.sort()

    all_data_points = []
    print(f"Tìm thấy {len(all_files)} file(s). Bắt đầu quét...")

    for file_path in all_files:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            # Lọc dữ liệu chỉ cho hostname mục tiêu
            df_filtered = df[df['hostname'] == TARGET_HOSTNAME]

            if df_filtered.empty:
                continue
            
            print(f"Tìm thấy {len(df_filtered)} bản ghi cho '{TARGET_HOSTNAME}' trong file {os.path.basename(file_path)}")

            for row in df_filtered.itertuples():
                timestamp = getattr(row, 'timestamp', None)
                int_info = getattr(row, 'int_info', None)
                pkt_loss_rate = getattr(row, 'pkt_loss_rate', None)
                
                if timestamp and int_info:
                    features = parse_dut11hw_int_info(int_info)
                    # Tạo nhãn dựa trên pkt_loss_rate
                    numeric_pkt_loss_rate = convert_value(pkt_loss_rate)
                    label = 1 if numeric_pkt_loss_rate > 0.01 else 0

                    if features:
                        all_data_points.append({
                            'timestamp': timestamp,
                            'features': features,
                            'label': label,
                            'source_file': os.path.basename(file_path)
                        })
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_path}: {e}")

    if not all_data_points:
        print(f"Không tìm thấy dữ liệu nào cho hostname '{TARGET_HOSTNAME}' trong tất cả các file.")
        exit()

    # Chuyển đổi thành DataFrame để sắp xếp theo thời gian
    processed_df = pd.DataFrame(all_data_points)
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    # Sắp xếp theo file trước, sau đó đến timestamp
    processed_df = processed_df.sort_values(by=['source_file', 'timestamp']).reset_index(drop=True)

    print(f"\nTổng hợp được {len(processed_df)} điểm dữ liệu.")

    # Trích xuất features và chuyển đổi sang tensor
    # .tolist() để chuyển từ Series của các list thành một list của các list
    feature_list = processed_df['features'].tolist()
    label_list = processed_df['label'].tolist()
    
    # Chuyển đổi sang tensor PyTorch
    features_tensor = torch.tensor(feature_list, dtype=torch.float32)
    labels_tensor = torch.tensor(label_list, dtype=torch.long)

    # Lưu tensor kết quả
    torch.save(
        {'features': features_tensor, 'labels': labels_tensor},
        OUTPUT_FILE
    )

    print("\n--- Hoàn tất ---")
    print(f"Đã lưu features và labels đã xử lý vào file: '{OUTPUT_FILE}'")
    print(f"Shape của tensor features: {features_tensor.shape}")
    print(f"Shape của tensor labels: {labels_tensor.shape}")
    print("\nMột vài mẫu features đầu tiên:")
    print(features_tensor[:10])
    print("\nMột vài mẫu labels đầu tiên:")
    print(labels_tensor[:10])
