# Bài 01 — Phân loại đa lớp hoa Iris với Keras

## Mục tiêu

Xây dựng mạng neural Keras dự đoán một bông hoa thuộc một trong ba loài: Setosa (`0`), Versicolor (`1`) hoặc Virginica (`2`). Dữ liệu Iris có 150 mẫu, mỗi mẫu gồm bốn số đo theo cm: chiều dài/rộng đài hoa và chiều dài/rộng cánh hoa.

## Nội dung thực hành

Chương trình `train_iris.py` thực hiện đầy đủ quy trình:

1. Đọc dữ liệu bằng `sklearn.datasets.load_iris()` và in thông tin ban đầu.
2. Chia 80% train, 20% test với `random_state=42` và `stratify=y`.
3. Chuẩn hóa bằng `StandardScaler`, chỉ `fit` trên train.
4. Huấn luyện mạng `4 → Dense(16, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)` trong 100 epoch.
5. Đánh giá loss, accuracy trên test và dự đoán một mẫu test.
6. Dùng `pandas.DataFrame(history.history)` theo phong cách khóa Kaggle để vẽ learning curves của train/validation loss và accuracy, lưu tại `iris_learning_loss.png` và `iris_learning_accuracy.png`.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
pip install -r keras-training/01-iris-multiclass/requirements.txt
python keras-training/01-iris-multiclass/train_iris.py
```

## Điểm cần nhớ

Đây là bài toán **đa lớp**. Lớp đầu ra gồm ba neuron softmax, ví dụ `[0.02, 0.93, 0.05]`. Xác suất lớn nhất ở vị trí `1`, nên dự đoán là Versicolor. Vì dùng `sparse_categorical_crossentropy`, nhãn đầu vào giữ dạng số nguyên `0`, `1`, `2`; không cần one-hot encoding.
