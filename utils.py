import torch
import torch.nn.functional as F

def reconstruction_loss(reconstructed, original):
    """
    Tính toán lỗi tái tạo (reconstruction loss) cho một batch.
    Sử dụng Mean Squared Error (MSE) làm hàm mất mát.

    Args:
        reconstructed (torch.Tensor): Dữ liệu được tái tạo bởi mô hình.
        original (torch.Tensor): Dữ liệu gốc.

    Returns:
        torch.Tensor: Một giá trị loss duy nhất cho cả batch.
    """
    return F.mse_loss(reconstructed, original)

def anomaly_score(reconstructed, original):
    """
    Tính toán điểm bất thường (anomaly score) cho mỗi chuỗi trong một batch.
    Điểm số là lỗi tái tạo trung bình (MSE) của mỗi chuỗi.

    Returns:
        torch.Tensor: Một tensor 1D chứa điểm bất thường cho mỗi chuỗi trong batch.
    """
    # Tính toán loss trên các chiều sequence và feature cho mỗi sample trong batch
    loss = torch.mean((reconstructed - original) ** 2, dim=[1, 2])
    return loss
