import matplotlib.pyplot as plt
import numpy as np

def drawAccuracyGraph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # 검증 정확도의 추세선 계산
    z = np.polyfit(epochs, val_acc, 1)  # 1차 다항식 (직선)으로 피팅
    p = np.poly1d(z)

    plt.figure(figsize=(14, 5))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.plot(epochs, p(epochs), 'r--', label='Validation acc trend')  # 추세선 추가
    plt.title('Training and validation accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# 예시 데이터로 history 객체를 정의해야 합니다.
class History:
    def __init__(self):
        self.history = {
            'accuracy': [0.1, 0.4, 0.6, 0.8],
            'val_accuracy': [0.2, 0.3, 0.5, 0.7],
            'loss': [2.0, 1.5, 1.0, 0.5],
            'val_loss': [2.2, 1.8, 1.3, 0.8]
        }

history = History()
drawAccuracyGraph(history)
