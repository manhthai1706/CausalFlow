# CausalFlow Architecture Overview

Tài liệu này mô tả cấu trúc kỹ thuật và quy trình vận hành của framework CausalFlow.

## 1. Sơ đồ hoạt động tổng thể (System Workflow)

Mô hình hoạt động dựa trên sự phối hợp giữa Deep Learning (MLP), Học đồ thị (NOTEARS) và Thống kê hạt nhân (GP/HSIC).

```mermaid
graph TD
    A[Input Data] --> B[Advanced Preprocessing]
    B --> B1[QuantileTransformer - Normalization]
    B --> B2[IsolationForest - Outlier Removal]
    
    subgraph CausalFlow_Core [CausalFlow Core Engine]
        C[Multivariate Backbone] --> D[VAE Head: Latent Discovery]
        C --> E[Neural Spline Flows: Noise Modeling]
        C --> F[NOTEARS: DAG Matrix Optimization]
        D & E & F --> G[Gaussian Process Head]
    end
    
    G --> H[Residual Extraction]
    H --> I[HSIC Independence Testing]
    I --> J[Causal Decision / DAG Matrix]
```

---

## 2. Chi tiết chức năng từng File

### Khối Module `core/` (Nền tảng thuật toán)

*   **`mlp.py` (Mạng nơ-ron đa nhiệm):**
    *   Sử dụng **ResBlocks** và **Attention** để trích xuất đặc trưng.
    *   **VAE Head:** Tìm kiếm các biến cơ chế tiềm ẩn (z).
    *   **Monotonic Spline Layer:** Triển khai Neural Spline Flows để xử lý nhiễu phi tuyến.
*   **`gppom_hsic.py` (Trái tim của mô hình):**
    *   Kết hợp toán học của NOTEARS (Ràng buộc đồ thị không vòng) với Gaussian Process.
    *   Tính toán h(W) penalty để ép ma trận trọng số về dạng DAG.
*   **`kernels.py`:** Thư viện nhân (RBF, Matern, Polynomial...) có tính đạo hàm để tối ưu hóa trực tiếp băng thông.
*   **`hsic.py`:** Triển khai các phép thử độc lập thống kê (Gamma Approximation và Permutation) để kiểm tra sự độc lập của phần dư.

### Khối Module `models/` (Giao diện cấp cao)

*   **`causalflow.py`:** Lớp bọc chính (Wrapper) cung cấp API `fit()`, `get_dag_matrix()` và `predict()` tương tự scikit-learn.
*   **`trainer.py`:** Quản lý vòng lặp huấn luyện, tối ưu hóa hàm toán tổng hợp (Likelihood + DAG Penalty + HSIC Penalty).
*   **`analysis.py`:** Xây dựng quy trình phân tích hướng nhân quả song biến (Bivariate) bằng cách so sánh độ độc lập phần dư giữa hai hướng giả định.

---

## 3. Quy trình phân tích hướng nhân quả (ANM-MM Flow)

Đây là quy trình giúp đạt được độ chính xác SOTA trên tập dữ liệu Sachs:

```mermaid
sequenceDiagram
    participant User as Người dùng
    participant ANM as Analysis Module
    participant CF as CausalFlow Instance
    
    User->>ANM: Gửi cặp biến (X, Y)
    ANM->>ANM: Tiền xử lý (Quantile + Isolation Forest)
    
    Note over ANM, CF: Thử nghiệm Hướng 1: X -> Y
    ANM->>CF: Khóa cấu trúc W_dag [0,1]=1
    CF-->>ANM: Trả về HSIC Score 1
    
    Note over ANM, CF: Thử nghiệm Hướng 2: Y -> X
    ANM->>CF: Khóa cấu trúc W_dag [1,0]=1
    CF-->>ANM: Trả về HSIC Score 2
    
    ANM->>ANM: So sánh (Score 1 vs Score 2)
    ANM-->>User: Kết quả hướng có HSIC thấp nhất
```

---

## 4. Hàm mất mát tổng hợp (Integrated Loss Function)

CausalFlow tối ưu hóa đồng thời 4 thành phần:
$$Loss = Loss_{Reg} + \alpha \cdot Loss_{DAG} + \beta \cdot \log(Loss_{HSIC}) + \gamma \cdot Loss_{KL}$$

1.  **$Loss_{Reg}$**: Sai số dự báo của Gaussian Process.
2.  **$Loss_{DAG}$**: Ràng buộc NOTEARS để đảm bảo cấu trúc là đồ thị không vòng.
3.  **$Loss_{HSIC}$**: Ép nhiễu (residuals) phải độc lập với nguyên nhân.
4.  **$Loss_{KL}$**: Ràng buộc phân phối cho việc khám phá biến ẩn.
