import torch
from torch import nn, optim
from tqdm import tqdm

from model import LstmAutoencoder
from get_train_data import get_train_dataloader
from utils import reconstruction_loss

# --- CẤU HÌNH HUẤN LUYỆN ---
N_EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_PATH = 'model.pt'

# Các tham số của mô hình (phải khớp với dữ liệu)
SEQ_LEN = 30      # Độ dài cửa sổ trượt
N_FEATURES = 4    # Số features (rxbs_xe1/1, rxbs_ge1/2, txbs_xe1/1, txbs_ge1/2)
EMBEDDING_DIM = 256 # Kích thước không gian tiềm ẩn
NUM_LAYERS = 3      # Số lớp LSTM
DROPOUT = 0.2       # Tỷ lệ dropout

def train_model():
    """
    Hàm chính để thực hiện quá trình huấn luyện mô hình.
    """
    # 1. Thiết lập thiết bị (GPU nếu có)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")  # Chỉ sử dụng CPU để tránh lỗi không tương thích với GPU
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải dữ liệu huấn luyện
    try:
        train_loader, _ = get_train_dataloader()
    except (FileNotFoundError, ValueError) as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return

    # 3. Khởi tạo mô hình, optimizer và loss function
    model = LstmAutoencoder(
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = reconstruction_loss # Sử dụng hàm loss từ utils.py

    print("\n--- Bắt đầu huấn luyện ---")

    # 4. Vòng lặp huấn luyện
    for epoch in range(N_EPOCHS):
        model.train() # Chuyển mô hình sang chế độ train
        
        total_loss = 0
        # Sử dụng tqdm để hiển thị thanh tiến trình
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS}", unit="batch")

        for inputs, targets in progress_bar:
            # Chuyển dữ liệu sang thiết bị đã chọn
            inputs, targets = inputs.to(device), targets.to(device)

            # Xóa các gradient cũ
            optimizer.zero_grad()

            # Forward pass: đưa dữ liệu qua mô hình
            reconstructed = model(inputs)

            # Tính toán loss
            loss = criterion(reconstructed, targets)

            # Backward pass: lan truyền ngược và cập nhật trọng số
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} - Loss trung bình: {avg_loss:.6f}")

    # 5. Lưu lại mô hình đã huấn luyện
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n--- Huấn luyện hoàn tất ---")
    print(f"Đã lưu mô hình vào file: '{MODEL_PATH}'")

if __name__ == '__main__':
    train_model()
