import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import joblib

# --- CẤU HÌNH ---
PROCESSED_DATA_FILE = 'dut11hw_processed_data.pt'
SCALER_FILE = 'scaler.gz'
SEQUENCE_LENGTH = 30  # Độ dài cửa sổ trượt, phải khớp với tham số của mô hình
BATCH_SIZE = 64       # Kích thước của mỗi batch dữ liệu

class TimeseriesDataset(Dataset):
    """
    Dataset tùy chỉnh của PyTorch để tạo các cửa sổ trượt từ dữ liệu chuỗi thời gian.
    Đối với Autoencoder, đầu vào và đầu ra (mục tiêu) là giống nhau.
    """
    def __init__(self, features, seq_len):
        self.features = features
        self.seq_len = seq_len

    def __len__(self):
        # Tổng số cửa sổ có thể tạo ra
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        # Lấy ra một cửa sổ (sequence)
        sequence = self.features[idx:idx + self.seq_len]
        # Trả về (đầu vào, mục tiêu). Với autoencoder, chúng là một.
        return sequence, sequence

def get_train_dataloader():
    """
    Tải dữ liệu đã xử lý, lọc dữ liệu bình thường, chuẩn hóa,
    và tạo ra một DataLoader sẵn sàng cho việc huấn luyện.
    
    Returns:
        tuple: (train_loader, scaler)
            - train_loader (DataLoader): DataLoader cho dữ liệu huấn luyện.
            - scaler (MinMaxScaler): Bộ chuẩn hóa đã được fit trên dữ liệu huấn luyện.
    """
    # 1. Tải dữ liệu đã xử lý
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(f"File dữ liệu '{PROCESSED_DATA_FILE}' không tồn tại. Vui lòng chạy 'timeseries_dataset.py' trước.")
    
    data = torch.load(PROCESSED_DATA_FILE)
    features = data['features']
    labels = data['labels']
    
    print(f"Đã tải dữ liệu với {features.shape[0]} điểm dữ liệu.")

    # 2. Lọc dữ liệu bình thường (label=0) để huấn luyện
    normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
    normal_features = features[normal_indices]
    
    if len(normal_features) < SEQUENCE_LENGTH:
        raise ValueError("Không đủ dữ liệu bình thường (label=0) để tạo ít nhất một chuỗi huấn luyện.")
        
    print(f"Tìm thấy {len(normal_features)} điểm dữ liệu bình thường để huấn luyện.")

    # 3. Chuẩn hóa dữ liệu feature
    # Quan trọng: Chỉ 'fit' scaler trên dữ liệu huấn luyện (dữ liệu bình thường)
    scaler = MinMaxScaler()
    scaled_normal_features = scaler.fit_transform(normal_features)
    
    # Lưu lại scaler để sử dụng sau này (ví dụ: trên tập test)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Đã fit và lưu scaler vào file '{SCALER_FILE}'.")
    
    # Chuyển dữ liệu đã chuẩn hóa về dạng tensor
    scaled_normal_features_tensor = torch.tensor(scaled_normal_features, dtype=torch.float32)

    # 4. Tạo Dataset với các cửa sổ trượt
    train_dataset = TimeseriesDataset(
        features=scaled_normal_features_tensor,
        seq_len=SEQUENCE_LENGTH
    )
    
    print(f"Đã tạo dataset huấn luyện với {len(train_dataset)} chuỗi.")

    # 5. Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Xáo trộn dữ liệu để huấn luyện tốt hơn
        num_workers=2, # Sử dụng worker để tải dữ liệu song song
        pin_memory=True
    )
    
    return train_loader, scaler

if __name__ == '__main__':
    # Ví dụ cách sử dụng hàm
    try:
        train_dataloader, _ = get_train_dataloader()
        
        print(f"\nĐã tạo DataLoader thành công.")
        print(f"Số lượng batch trong DataLoader: {len(train_dataloader)}")
        
        # Lấy một batch để kiểm tra
        for inputs, targets in train_dataloader:
            print(f"Shape của một batch đầu vào: {inputs.shape}")
            print(f"Shape của một batch mục tiêu: {targets.shape}")
            break # Chỉ kiểm tra batch đầu tiên
            
    except (FileNotFoundError, ValueError) as e:
        print(f"\nLỗi: {e}")
