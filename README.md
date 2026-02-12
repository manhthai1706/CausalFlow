# CausalFlow

[![Architecture](https://img.shields.io/badge/Kiến_trúc-Chi_tiết-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Tổng quan

CausalFlow là một thư viện Python dùng để xác định mối quan hệ nhân quả từ dữ liệu quan sát. Dự án được phát triển dựa trên framework ANM-MM (Additive Noise Model - Mixture Model) của [amber0309](https://github.com/amber0309/ANM-MM), với các thay đổi chính ở phần backbone mạng nơ-ron và cách tổ chức mã nguồn.

Mục tiêu của dự án là gói gọn toàn bộ quy trình khám phá nhân quả — từ tiền xử lý dữ liệu, huấn luyện mô hình đến trích xuất đồ thị — vào trong một class `CausalFlow` duy nhất, thay vì phải gọi nhiều hàm rời rạc như bản gốc.

## Giới thiệu

Bài toán khám phá nhân quả (Causal Discovery) đặt câu hỏi: cho hai biến X và Y có tương quan, liệu X gây ra Y, Y gây ra X, hay cả hai đều do một biến ẩn thứ ba chi phối? Đây là vấn đề cơ bản trong nhiều lĩnh vực như y sinh, kinh tế và khoa học xã hội.

CausalFlow tiếp cận bài toán này theo hướng mô hình nhiễu cộng (Additive Noise Model): nếu Y = f(X) + N với N độc lập với X, thì X là nguyên nhân của Y. Mô hình sử dụng mạng nơ-ron để xấp xỉ hàm f, sau đó kiểm tra tính độc lập giữa phần dư và biến đầu vào bằng HSIC (Hilbert-Schmidt Independence Criterion).

## Các thành phần kỹ thuật

CausalFlow sử dụng các thành phần sau:

### Backbone mạng nơ-ron
- **ResNet blocks**: Các khối residual giúp huấn luyện mạng sâu hơn mà không bị vanishing gradient.
- **Gated Residual Network (GRN)**: Cơ chế gating để kiểm soát luồng thông tin, lấy ý tưởng từ Temporal Fusion Transformers.
- **Self-Attention**: Lớp attention đơn giản để đánh trọng số các biến đầu vào.

### Mô hình hóa nhiễu
- **Neural Spline Flows (NSF)**: Dùng các hàm spline đơn điệu để biến đổi phân phối nhiễu, thay vì giả định nhiễu tuân theo phân phối cố định (ví dụ Gaussian).
- **VAE head**: Mã hóa biến tiềm ẩn (latent variable) đại diện cho các cơ chế nhân quả khác nhau trong mô hình hỗn hợp.

### Tối ưu hóa cấu trúc đồ thị
- **NOTEARS**: Phương pháp tối ưu hóa liên tục để học ma trận kề W của đồ thị nhân quả, với ràng buộc đại số đảm bảo đồ thị không có vòng.
- **HSIC penalty**: Hàm phạt dựa trên HSIC để ép phần dư và biến nguyên nhân độc lập về mặt thống kê.

### Tiền xử lý dữ liệu
- **QuantileTransformer** (scikit-learn): Chuẩn hóa phân phối dữ liệu về dạng Gaussian.
- **Isolation Forest** (scikit-learn): Loại bỏ các điểm ngoại lai trước khi đưa vào mô hình.

## So sánh với dự án gốc (amber0309)

| Thành phần | amber0309 (Base) | CausalFlow |
| :--- | :--- | :--- |
| Cấu trúc mã | Các script và hàm riêng lẻ | Tổ chức theo package (core/models/utils) |
| Backbone | MLP tiêu chuẩn | ResNet + GRN + Attention |
| Mô hình nhiễu | Phân phối cố định | Neural Spline Flows |
| Phạm vi | Chủ yếu song biến | Hỗ trợ đa biến qua NOTEARS |
| Giao diện | Gọi hàm trực tiếp | Class API (`model.fit()`, `model.predict_direction()`) |
| Phân tích thêm | Không | Counterfactual, stability check |
| Tiền xử lý | Thủ công | Tích hợp IsolationForest + QuantileTransformer |

## Cài đặt

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

## Sử dụng

### Xác định hướng nhân quả (song biến)
```python
from causalflow import CausalFlow

model = CausalFlow(lda=12.0)
direction = model.predict_direction(data)  # 1: X->Y, -1: Y->X
```

### Huấn luyện đa biến
```python
model = CausalFlow(data=X, epochs=200)
W_raw, W_binary = model.get_dag_matrix()
```

## Kết quả Thực nghiệm

Đánh giá trên tập dữ liệu Sachs (Protein Signaling Network), gồm 11 biến protein và 17 cạnh nhân quả đã biết:

- **Accuracy**: 70.6% (12/17 cạnh đúng hướng).
- **SHD**: 5.

### So sánh hiệu năng

| Phương pháp | Accuracy (Sachs) | SHD |
| :--- | :--- | :--- |
| PC Algorithm | ~50-55% | Cao |
| NOTEARS (gốc) | ~60% | > 8 |
| CausalFlow | 70.6% | 5 |

## Tham khảo

1. **ANM-MM (amber0309).** [GitHub](https://github.com/amber0309/ANM-MM).
2. **Zheng, X., et al. (2018).** "DAGs with NO TEARS." *NeurIPS*.
3. **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*.
4. **Zhang, K., & Hyvarinen, A. (2009).** "On the Identifiability of the Post-Nonlinear Causal Model." *UAI*.
5. **Rahimi, A., & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
6. **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*.
7. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.
8. **Jang, E., et al. (2016).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR*.
9. **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*.
10. **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*.
11. **Ba, J. L., et al. (2016).** "Layer Normalization." *arXiv*.
12. **Hendrycks, D., & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv*.
13. **Lim, B., et al. (2021).** "Temporal Fusion Transformers." *IJF*.
14. **Loshchilov, I., & Hutter, F. (2017).** "Decoupled Weight Decay Regularization." *ICLR*.
15. **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*.
16. **Pedregosa, F., et al. (2011).** "Scikit-learn." *JMLR*.
17. **Paszke, A., et al. (2019).** "PyTorch." *NeurIPS*.

## License
MIT License.
