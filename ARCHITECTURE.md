# CausalFlow: Detailed File-by-File Architecture

Tài liệu này cung cấp sơ đồ hoạt động chi tiết cho từng thành phần trong mã nguồn của framework CausalFlow.

---

## 📂 Thư mục `causalflow/core/` (Nền tảng thuật toán)

### 1. `mlp.py` - Ultimate Deep Learning Backbone
Đây là tệp phức tạp nhất, chịu trách nhiệm trích xuất thực thể và mô hình hóa nhiễu.

```mermaid
graph TD
    IN[Input X] --> ATT[Attention Layer: Feature Selection]
    ATT --> GRN[Gated Residual Network: GRN]
    GRN --> RB[ResBlocks: Residual Learning]
    
    subgraph Multi-Head_Outputs
        RB --> VAE[VAE Head: mu, log_var for Mechanism Z]
        RB --> NSF[Monotonic Spline: Noise Transformation h_y]
        RB --> REG[Regressor: Probabilistic Output mu_y, var_y]
    end
    
    VAE --> Z[Softmax Z clusters]
    NSF --> HY[Y Transformation]
```

### 2. `gppom_hsic.py` - Core Engine & DAG Learning
Điều phối việc học đồ thị nhân quả và kết hợp các hàm mất mát.

```mermaid
graph TD
    B[Batch Data] --> MLP[Call: mlp.py for Latents]
    MLP --> Z[Mechanism Z]
    
    subgraph DAG_Optimization
        W[W_dag Matrix] --> PEN[Acyclicity Penalty: h_W]
        W --> MASK[Structural Masking]
    end
    
    subgraph Prediction_Flow
        B & MASK --> GP[Random Fourier Features GP]
        GP --> PRED[Y Prediction]
    end
    
    PRED --> MSE[Loss: Regression]
    Z & B --> HSIC1[Loss: FastHSIC Clustering]
    PRED & B --> HSIC2[Loss: Adaptive HSIC PNL]
    
    MSE & PEN & HSIC1 & HSIC2 --> TOTAL[Total Loss & Backward]
```

### 3. `hsic.py` - Statistical Independence Testing
Triển khai các phép thử thống kê để xác nhận quan hệ nhân quả.

```mermaid
graph LR
    subgraph hsic_gam
        A[Data X, Y] --> K[Compute Kernels K, L]
        K --> H[Trace Calculation]
        H --> GAM[Gamma Approximation]
        GAM --> P[p-value / Stat]
    end
    
    subgraph hsic_perm
        A1[Data] --> K1[Kernels]
        K1 --> SHUFFLE[Permutation/Shuffle]
        SHUFFLE --> DIST[Null Distribution]
    end
```

### 4. `kernels.py` - Differentiable Kernel Library
Sơ đồ phân cấp các hàm nhân có thể đạo hàm.

```mermaid
graph TD
    K[Base Kernel] --> RBF[RBF / Gaussian]
    K --> MAT[Matern 3/2 & 5/2]
    K --> RQ[Rational Quadratic]
    K --> LIN[Linear / Poly]
    
    subgraph Optimization
        PARAM[log_alpha, log_gamma] --> GRAD[Learnable via SGD]
    end
```

---

## 📂 Thư mục `causalflow/models/` (Giao diện & Ứng dụng)

### 5. `causalflow.py` - Sklearn-style Wrapper
Giao diện chính cho người dùng cuối.

```mermaid
graph TD
    START[CausalFlow Object] --> INIT[Init Dimensions & Device]
    INIT --> FIT[Method: fit]
    
    subgraph FIT_Logic
        FIT --> BIV[Check: Bivariate X, Y?]
        FIT --> MULTI[Check: Multivariate X?]
        BIV & MULTI --> TRAIN[Create: CausalFlowTrainer]
    end
    
    TRAIN --> RESULT[Update History & Weights]
    RESULT --> DAG[Method: get_dag_matrix]
```

### 6. `trainer.py` - Training Orchestrator
Quản lý vòng lặp huấn luyện và lịch trình (scheduling).

```mermaid
graph TD
    LOOP[For Epoch in Epochs] --> TEMP[Adjust Temperature: Gumbel-Softmax]
    TEMP --> BATCH[For Batch in DataLoader]
    
    subgraph Batch_Processing
        BATCH --> ZERO[optimizer.zero_grad]
        ZERO --> FORWARD[model.forward]
        FORWARD --> BACK[loss.backward]
        BACK --> STEP[optimizer.step]
    end
    
    STEP --> LOG[Logging: Loss & HSIC Trend]
```

### 7. `analysis.py` - Causal Direction Discovery
Lô-gic phân tích nhân quả nâng cao (SOTA 70.6%).

```mermaid
graph TD
    DATA[Raw Data Pair] --> PRE[Standardize / Quantile Transform]
    PRE --> CLEAN[Isolation Forest: Remove Outliers]
    
    subgraph Hypothesis_Testing
        CLEAN --> H1[Test Hypothesis: X -> Y]
        H1 --> LOCK1[Lock W_dag: Force Direction]
        LOCK1 --> SCORE1[Compute HSIC Stability Score 1]
        
        CLEAN --> H2[Test Hypothesis: Y -> X]
        H2 --> LOCK2[Lock W_dag: Force Direction]
        LOCK2 --> SCORE2[Compute HSIC Stability Score 2]
    end
    
    SCORE1 & SCORE2 --> COMP[Compare Scores]
    COMP --> DECIDE[Final Decision: Min Score Wins]
```
