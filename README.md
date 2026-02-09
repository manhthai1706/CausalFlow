# CausalFlow: Advanced Multivariate Causal Discovery Framework

## Overview
CausalFlow is a deep learning framework designed for multivariate causal discovery and counterfactual reasoning. It integrates Gaussian Processes, Hilbert-Schmidt Independence Criterion (HSIC), and modern generative architectures to identify causal structures within complex datasets.

## Acknowledgments
This framework is built upon the foundational research and implementation of the GPPOM-HSIC methodology by [amber0309](https://github.com/amber0309). We express our sincere gratitude for their contributions to the field of causal discovery, which served as the base for this advanced SOTA version.

## Installation

### 1. Cài đặt trực tiếp từ GitHub (Dành cho người dùng)
Người khác có thể cài đặt thư viện của bạn chỉ bằng một dòng lệnh mà không cần tải source code về thủ công:

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

### 2. Cài đặt từ Source Code (Dành cho nhà phát triển)
Nếu muốn chỉnh sửa code, họ có thể clone repo và cài đặt:

```bash
git clone https://github.com/manhthai1706/CausalFlow.git
cd CausalFlow
pip install -e .
```

## Quick Start (Cách sử dụng)

Sau khi cài đặt, việc sử dụng sẽ giống hệt như các thư viện phổ biến (`numpy`, `sklearn`):

```python
import numpy as np
from causalflow import CausalFlow, ANMMM_cd_advanced

# 1. Khai phá cấu trúc DAG đa biến
data = np.random.randn(1000, 5) # Ví dụ 5 biến
model = CausalFlow(x_dim=5, n_clusters=3)
model.fit(data, epochs=100)

# Lấy ma trận kề (Adjacency Matrix)
W_raw, W_binary = model.get_dag_matrix(threshold=0.1)
print("Learned DAG Structure:\n", W_binary)

# 2. Kiểm tra hướng nhân quả giữa 2 biến (Bivariate)
pair_data = data[:, :2]
direction, analyzer = ANMMM_cd_advanced(pair_data, lda=10.0)
# direction = 1 (X->Y) hoặc -1 (Y->X)
```

## Requirements
Hệ thống sẽ tự động cài đặt các thư viện sau:
- PyTorch >= 2.0.0
- NumPy
- Scikit-learn
- Scipy
- Matplotlib

## License
MIT License
