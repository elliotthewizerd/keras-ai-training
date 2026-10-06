"""Huấn luyện mạng Keras phân loại 3 loài hoa Iris."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# 1. Đọc dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target
species = iris.target_names
print("Kích thước X:", X.shape)
print("Kích thước y:", y.shape)
print("5 mẫu đầu tiên (X):\n", X[:5])
print("5 nhãn đầu tiên (y):", y[:5])

# 2. Chia 80% train và 20% test, giữ nguyên tỉ lệ từng loài.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chỉ fit scaler trên train để tránh rò rỉ thông tin từ test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Mạng phân loại đa lớp: 4 -> 16 -> 8 -> 3.
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax"),
])

# 5. Nhãn là số nguyên 0, 1, 2 nên dùng sparse categorical loss.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 6. validation_split lấy 20% từ train, không sử dụng test.
history = model.fit(
    X_train_scaled, y_train, epochs=100, batch_size=16,
    validation_split=0.2, verbose=1,
)

# Learning curves của train và validation.
epochs = range(1, len(history.history["loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, history.history["loss"], label="Train loss")
axes[0].plot(epochs, history.history["val_loss"], label="Validation loss")
axes[0].set(title="Learning curve: Loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(epochs, history.history["accuracy"], label="Train accuracy")
axes[1].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
axes[1].set(title="Learning curve: Accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iris_learning_curves.png", dpi=150)
plt.show()

# 7. Đánh giá cuối cùng trên tập test.
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 8. Dự đoán một mẫu thuộc test.
sample_index = 0
probabilities = model.predict(X_test_scaled[sample_index:sample_index + 1], verbose=0)[0]
predicted_label = int(np.argmax(probabilities))
print("\nMẫu test được chọn:", X_test[sample_index])
print("Nhãn thật:", int(y_test[sample_index]), "-", species[y_test[sample_index]].title())
print("Xác suất từng loài:")
for label, probability in enumerate(probabilities):
    print(f"  {label} - {species[label].title()}: {probability:.4f}")
print("Loài mô hình dự đoán:", predicted_label, "-", species[predicted_label].title())
