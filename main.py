from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([Flatten(input_shape=(28, 28)), Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

index = np.random.randint(0, len(x_test))
sample = x_test[index].reshape(1, 28, 28)
prediction = model.predict(sample)
predicted_label = np.argmax(prediction)

# 9. Show the image and prediction
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()