import torch
from torch import nn
import math

class LstmAutoencoder(nn.Module):
    """
    Một mô hình LSTM Autoencoder mạnh mẽ hơn, sử dụng:
    - Encoder đa lớp, hai chiều (bidirectional) để nắm bắt ngữ cảnh từ cả hai hướng.
    - Decoder đa lớp, một chiều để tái tạo lại chuỗi.
    - Dropout để chống lại việc học vẹt (overfitting).
    """
    def __init__(self, seq_len, n_features, embedding_dim=128, num_layers=2, dropout=0.2):
        """
        Args:
            seq_len (int): Độ dài của chuỗi đầu vào/đầu ra.
            n_features (int): Số lượng feature. Dựa trên `timeseries_dataset.py`, giá trị này là 4.
            embedding_dim (int): Kích thước của không gian tiềm ẩn (hyperparameter).
            num_layers (int): Số lượng lớp LSTM xếp chồng lên nhau.
            dropout (float): Tỷ lệ dropout giữa các lớp LSTM (nếu num_layers > 1).
        """
        super(LstmAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Encoder: Đa lớp, hai chiều, có dropout
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder: Đa lớp, một chiều, có dropout
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Lớp Linear để chuyển đổi output của LSTM về đúng số chiều của feature ban đầu
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        """
        Thực hiện quá trình mã hóa và giải mã.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có shape (batch_size, seq_len, n_features).
        
        Returns:
            torch.Tensor: Tensor tái tạo có shape (batch_size, seq_len, n_features).
        """
        # Mã hóa
        # Trạng thái ẩn của encoder hai chiều có shape: (num_layers * 2, batch_size, embedding_dim)
        _, (hidden, cell) = self.encoder(x)

        # Xử lý trạng thái ẩn và ô để phù hợp với decoder một chiều
        # Gộp trạng thái của chiều xuôi và ngược bằng cách cộng chúng lại
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.embedding_dim).sum(dim=1)
        cell = cell.view(self.num_layers, 2, batch_size, self.embedding_dim).sum(dim=1)
        
        # Chuẩn bị đầu vào cho decoder
        # Lấy trạng thái ẩn của lớp cuối cùng làm vector ngữ cảnh
        context_vector = hidden[-1, :, :]
        # Lặp lại vector ngữ cảnh `seq_len` lần để tạo thành chuỗi đầu vào cho decoder
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Giải mã
        reconstruction, _ = self.decoder(decoder_input, (hidden, cell))
        # Chuyển đổi output của decoder về đúng số chiều feature ban đầu
        reconstruction = self.output_layer(reconstruction)
        return reconstruction


if __name__ == '__main__':
    # --- Ví dụ sử dụng ---
    # Các tham số này cần khớp với dữ liệu của bạn
    SEQ_LEN = 30
    N_FEATURES = 4
    EMBEDDING_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # Khởi tạo mô hình
    model = LstmAutoencoder(
        seq_len=SEQ_LEN, 
        n_features=N_FEATURES, 
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    print("Cấu trúc mô hình:")
    print(model)

    # Tạo dữ liệu giả để kiểm tra
    batch_size = 16
    dummy_input = torch.randn(batch_size, SEQ_LEN, N_FEATURES)
    print(f"\nShape của input giả: {dummy_input.shape}")

    # Chạy thử mô hình
    reconstructed_output = model(dummy_input)
    print(f"Shape của output tái tạo: {reconstructed_output.shape}")

    # Kiểm tra shape có khớp không
    assert dummy_input.shape == reconstructed_output.shape
    print("\nKiểm tra shape thành công! Mô hình hoạt động đúng.")


# --- Transformer Autoencoder Model ---

class PositionalEncoding(nn.Module):
    """
    Thêm thông tin về vị trí của các token trong chuỗi.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    Mô hình Autoencoder dựa trên kiến trúc Transformer.
    """
    def __init__(self, seq_len, n_features, embedding_dim=128, n_heads=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        """
        Args:
            seq_len (int): Độ dài chuỗi.
            n_features (int): Số lượng features.
            embedding_dim (int): Kích thước embedding (d_model).
            n_heads (int): Số lượng attention heads.
            num_encoder_layers (int): Số lớp trong encoder.
            num_decoder_layers (int): Số lớp trong decoder.
            dim_feedforward (int): Kích thước lớp feedforward.
            dropout (float): Tỷ lệ dropout.
        """
        super(TransformerAutoencoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(n_features, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=seq_len)

        # Định nghĩa Encoder Layer và Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Định nghĩa Decoder Layer và Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Lớp linear cuối cùng để đưa về đúng số chiều feature ban đầu
        self.output_projection = nn.Linear(embedding_dim, n_features)

    def forward(self, src):
        """
        Args:
            src (torch.Tensor): Tensor đầu vào có shape (batch_size, seq_len, n_features).
        
        Returns:
            torch.Tensor: Tensor tái tạo có shape (batch_size, seq_len, n_features).
        """
        # Chuyển đổi input và thêm positional encoding
        src = self.input_projection(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)

        # Mã hóa
        memory = self.transformer_encoder(src)
        
        # Giải mã
        # Trong autoencoder, đầu vào của decoder (tgt) có thể chính là đầu vào của encoder
        output = self.transformer_decoder(src, memory)
        
        # Đưa về số chiều ban đầu
        output = self.output_projection(output)
        return output

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Kiểm tra mô hình Transformer Autoencoder")
    print("="*50)
    
    # Các tham số cho mô hình Transformer
    model_tf = TransformerAutoencoder(
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        embedding_dim=128,
        n_heads=4 # embedding_dim phải chia hết cho n_heads
    )
    print("Cấu trúc mô hình Transformer:")
    print(model_tf)

    reconstructed_output_tf = model_tf(dummy_input)
    print(f"\nShape của output tái tạo từ Transformer: {reconstructed_output_tf.shape}")
    assert dummy_input.shape == reconstructed_output_tf.shape
    print("\nKiểm tra shape thành công! Mô hình Transformer hoạt động đúng.")