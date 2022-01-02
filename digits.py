from sklearn import datasets

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import matplotlib.pyplot as plt
from NN import NN_classifier

# Load the digits dataset
digits = datasets.load_digits()

n_train = 1400

model = NN_classifier([40, 30, 20, 20, 20])

x = np.array( [np.concatenate(np.array(image / 15)) for image in digits.images] )

y = []
for t in digits.target:
    target_v = [0 for i in range(10)]
    target_v[t] = 1
    y.append(target_v)
np.array(y)

model = NN_classifier(hls = [40, 30, 20, 20, 20], lr = 1).set_train_data(x[:n_train], y[:n_train])
model.set_init_params()

#model = model.load_model('NN_params')

for _ in range(20):
    model.train(n_epoch= 5, decay = 0.99)
    c = 0
    i = 0
    for img in x[(n_train + 1):-1]:
        if y[i][model.get_max_idx(x[i])] == 1:
            c += 1
        i += 1
    print(f"Test accuracy: {round(c/i*100, 3)}% ({len(x[(n_train + 1):-1])} images in the test set)")

model = model.save_model('NN_params')

# Display a digit
for i in range(n_train + 1, n_train + 100):
    idx = model.get_max_idx(x[i])
    print(f"is this a {idx}? {'YES!' if y[i][idx] == 1 else 'NO :('}")
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()