# CausalFlow: Advanced Multivariate Causal Discovery Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CausalFlow** là một framework học sâu chuyên sâu dành cho việc khám phá cấu trúc nhân quả đa biến (multivariate causal discovery) và phân tích phản thực tế (counterfactual reasoning). Framework này tích hợp các kỹ thuật tiên tiến nhất để giải quyết các bài toán nhân quả trong môi trường nhiễu phi Gaussian và quan hệ phi tuyến phức tạp.

## 🚀 Tính năng nổi bật (SOTA Architecture)

- **Neural Spline Flows (NSF):** Sử dụng các hàm Spline bậc ba có tính đơn điệu để mô hình hóa nhiễu với độ linh hoạt cực cao, vượt xa các phương pháp Affine Coupling truyền thống.
- **Differentiable DAG Learning (NOTEARS):** Tối ưu hóa cấu trúc đồ thị nhân quả thông qua ràng buộc tính không vòng (acyclicity) liên tục, cho phép học trực tiếp bằng Gradient Descent.
- **Variational Latent Discovery:** Sử dụng cấu trúc VAE để phát hiện các cơ chế tiềm ẩn và giảm thiểu tác động của các biến ẩn (unobserved confounders).
- **Adaptive HSIC Regularization:** Tối ưu hóa băng thông kernel tự động để thực hiện các phép thử độc lập thống kê với độ nhạy tối đa.
- **Post-Nonlinear (PNL) Recovery:** Khả năng khôi phục hướng nhân quả ngay cả khi dữ liệu trải qua các biến đổi phi tuyến sau nhiễu.

## 📦 Cài đặt

Cài đặt trực tiếp từ tài nguyên hệ thống (Recommended):

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

Hoặc cài đặt chuyên sâu cho môi trường nghiên cứu:

```bash
git clone https://github.com/manhthai1706/CausalFlow.git
cd CausalFlow
pip install -e .
```

## 🛠 Hướng dẫn sử dụng chuyên sâu

### 1. Phân tích hướng nhân quả Song biến (Bivariate Discovery)
Sử dụng mô hình hỗn hợp ANM-MM nâng cao để xác định hướng $X \rightarrow Y$ hoặc $Y \rightarrow X$:

```python
from causalflow import ANMMM_cd_advanced
import numpy as np

# Giả sử 'data' là ma trận [N, 2] đã được chuẩn hóa
# lda: Tham số điều chỉnh trọng số HSIC (Thường từ 10.0 - 20.0)
direction, analyzer = ANMMM_cd_advanced(data, lda=15.0)

if direction == 1:
    print("Mô hình xác nhận: X là nguyên nhân của Y")
else:
    print("Mô hình xác nhận: Y là nguyên nhân của X")
```

### 2. Khám phá cấu trúc đồ thị Đa biến (Multivariate DAG)
Học toàn bộ ma trận kề cho hệ thống nhiều biến số:

```python
from causalflow import CausalFlow

# Khởi tạo mô hình cho 11 biến (Ví dụ: Sachs dataset)
model = CausalFlow(x_dim=11, n_clusters=3, lda=1.0)

# Huấn luyện tích hợp ràng buộc NOTEARS và Spline Flows
model.fit(data, epochs=200, batch_size=64)

# Trích xuất ma trận cấu hình (Adjacency Matrix)
W_raw, W_binary = model.get_dag_matrix(threshold=0.1)
```

## 📊 Hiệu suất thực nghiệm (Benchmarks)

Trên bộ dữ liệu sinh học **Sachs (Flow Cytometry)**, CausalFlow đạt độ chính xác hướng nhân quả **47.1%** (8/17 cạnh chính xác) chỉ với dữ liệu quan sát thuần túy. Đây là kết quả vượt trội so với các thuật quy trình PC (30%) hoặc GES (23%) truyền thống trên cùng một tập dữ liệu.

## 🎓 Tri ân & Tham khảo

Dự án này được phát triển dựa trên nền tảng nghiên cứu và mã nguồn gốc GPPOM-HSIC của tác giả [amber0309](https://github.com/amber0309). Chúng tôi chân thành cảm ơn những đóng góp quan trọng của tác giả đối với cộng đồng nghiên cứu nhân quả.

- Zheng, X., et al. "DAGs with NO TEARS" (2018).
- Durkan, C., et al. "Neural Spline Flows" (2019).
- Gretton, A., et al. "Kernel Independence Tests" (2007).

## 📄 License
Tài nguyên này được phát hành dưới giấy phép **MIT License**.
