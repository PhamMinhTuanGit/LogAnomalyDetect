import torch
import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from model import LstmAutoencoder
from utils import anomaly_score

# --- CẤU HÌNH ---
MODEL_PATH = 'model.pt'
SCALER_FILE = 'scaler.gz'
PROCESSED_DATA_FILE = 'dut11hw_processed_data.pt'
SEQ_LEN = 30
BATCH_SIZE = 128 # Có thể dùng batch size lớn hơn trong quá trình inference

# Các tham số của mô hình (phải khớp với mô hình đã huấn luyện)
N_FEATURES = 4
EMBEDDING_DIM = 256 # Kích thước không gian tiềm ẩn
NUM_LAYERS = 3     # Số lớp LSTM
DROPOUT = 0.2       # Tỷ lệ dropout

class InferenceDataset(Dataset):
    """
    Dataset tùy chỉnh để tạo cửa sổ trượt cho quá trình inference.
    Nó cũng trả về nhãn tương ứng với mỗi cửa sổ.
    """
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.seq_len]
        # Nhãn của một chuỗi thường được quyết định bởi điểm cuối cùng của chuỗi đó
        label = self.labels[idx + self.seq_len - 1]
        return sequence, label

def find_anomaly_threshold(model, normal_loader, device, percentile=90):
    """
    Tìm ngưỡng bất thường dựa trên lỗi tái tạo của dữ liệu bình thường.
    """
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for sequences, _ in tqdm(normal_loader, desc="Finding Threshold"):
            sequences = sequences.to(device)
            reconstructed = model(sequences)
            scores = anomaly_score(reconstructed, sequences)
            reconstruction_errors.extend(scores.cpu().numpy())
            
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold

def plot_reconstruction_errors(errors_normal, errors_anomaly, threshold):
    """
    Vẽ biểu đồ phân phối của lỗi tái tạo cho dữ liệu bình thường và bất thường.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(errors_normal, bins=50, kde=True, label='Lỗi (Bình thường)', color='blue')
    sns.histplot(errors_anomaly, bins=50, kde=True, label='Lỗi (Bất thường)', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Ngưỡng ({threshold:.6f})')
    plt.title('Phân phối của Lỗi Tái tạo')
    plt.xlabel('Lỗi Tái tạo (MSE)')
    plt.ylabel('Tần suất')
    plt.legend()
    plt.show()

def inference():
    """
    Hàm chính để thực hiện quá trình inference và đánh giá.
    """
    # 1. Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải mô hình và scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError("Không tìm thấy file model hoặc scaler. Vui lòng chạy 'train.py' trước.")
        
    model = LstmAutoencoder(
        seq_len=SEQ_LEN, 
        n_features=N_FEATURES, 
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    scaler = joblib.load(SCALER_FILE)

    # 3. Tải và chuẩn bị dữ liệu
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu '{PROCESSED_DATA_FILE}'. Vui lòng chạy 'timeseries_dataset.py' trước.")
        
    data = torch.load(PROCESSED_DATA_FILE)
    features = data['features']
    labels = data['labels']

    # Chuẩn hóa toàn bộ features bằng scaler đã lưu
    scaled_features = scaler.transform(features.numpy())
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # 4. Tạo DataLoader cho dữ liệu bình thường để tìm ngưỡng
    normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
    normal_features_tensor = scaled_features_tensor[normal_indices]
    
    # Sử dụng lại class InferenceDataset, nhãn không quan trọng ở bước này
    normal_dataset = InferenceDataset(normal_features_tensor, torch.zeros_like(normal_indices), SEQ_LEN)
    normal_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Tìm ngưỡng bất thường
    print("\n--- Xác định ngưỡng bất thường trên dữ liệu bình thường ---")
    threshold = find_anomaly_threshold(model, normal_loader, device)
    print(f"Ngưỡng bất thường được xác định là: {threshold:.6f}")

    # 6. Tạo DataLoader cho toàn bộ dữ liệu để đánh giá và trực quan hóa
    full_dataset = InferenceDataset(scaled_features_tensor, labels, SEQ_LEN)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Lấy điểm lỗi và nhãn cho toàn bộ dữ liệu
    print("\n--- Tính toán lỗi tái tạo trên toàn bộ dữ liệu ---")
    model.eval()
    all_scores = []
    all_true_labels = []
    with torch.no_grad():
        for sequences, labels_batch in tqdm(full_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            reconstructed = model(sequences)
            scores = anomaly_score(reconstructed, sequences)
            all_scores.extend(scores.cpu().numpy())
            all_true_labels.extend(labels_batch.numpy())
    
    all_scores = np.array(all_scores)
    all_true_labels = np.array(all_true_labels)

    # 8. Trực quan hóa lỗi
    normal_scores_from_eval = all_scores[all_true_labels == 0]
    anomaly_scores_from_eval = all_scores[all_true_labels == 1]
    plot_reconstruction_errors(normal_scores_from_eval, anomaly_scores_from_eval, threshold)

    # In báo cáo phân loại
    predictions = (all_scores > threshold).astype(int)
    print("\n--- Báo cáo phân loại chi tiết ---")
    print(classification_report(
        all_true_labels, 
        predictions, 
        target_names=['Normal (0)', 'Anomaly (1)'],
        zero_division=0
    ))

    # Tính và in riêng các chỉ số quan trọng
    accuracy = accuracy_score(all_true_labels, predictions)
    # Tính F1-score cho lớp bất thường (positive class)
    f1 = f1_score(all_true_labels, predictions, pos_label=1, zero_division=0)

    print("\n--- Tóm tắt hiệu suất ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (cho lớp Anomaly): {f1:.4f}")

if __name__ == '__main__':
    try:
        inference()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nLỗi: {e}")
