import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import matplotlib.pyplot as plt
import pickle

class NN_interpolation:
    def __init__(self, hls, n_input = 1, n_out = 1, cost_metric = "L1"):
        lrs = [n_input]
        for l in hls:
            lrs.append(l)
        lrs.append(n_out)

        self.lrs = lrs
        self.n_input = n_input
        self.n_output = n_out
        self.set_params()

        self.cost_metric = cost_metric

    def set_params(self):
        params = []
        for i in range(len(self.lrs) - 1):
            W = self.set_weights(np.random.rand(self.lrs[i+1], self.lrs[i]), -1, 1)
            b = self.set_weights(np.random.rand(self.lrs[i+1],1), -5, 5)
            params.append( (W, b) )
        self.p = params
        return self

    def eval(self, x, p = None):
        if self and not p:
            p = self.p

        x = x.reshape(-1, 1)
        # hidden layers
        for W, b in p[:-1]:
            x = np.tanh(np.dot(W, x) + b)
            #print(x)

        # output layer
        W, b = p[-1]
        y = np.dot(W, x) + b

        return y
    
    def cost(self, x, y, p = None):
        y_hat = NN_interpolation.eval(self, x, p)
        e = y_hat - y
        if self.cost_metric == "L1":
            e = np.abs(e)
        elif self.cost_metric == "SQ":
            e = e ** 2
        else:
            print("Cost metric not defined.")
        return np.sum(e)

    def set_train_data(self, x, y):
        N_samples_x = len(x) if hasattr(x, '__len__') else 1
        N_samples_y = len(y) if hasattr(y, '__len__') else 1
        if N_samples_x != N_samples_y:
            print("Sample number does not match for x and y.")
            return
        self.N_samples = N_samples_x

        N_outputs = len(y[0]) if hasattr(y[0], '__len__') else 1
        n_input = len(x[0]) if hasattr(x[0], '__len__') else 1

        if self.n_input != n_input or self.n_output != N_outputs:
            self.n_input = n_input
            self.n_output = N_outputs
            self.lrs[0] = n_input
            self.lrs[-1] = N_outputs
            print("Updated input and output layers sizes.")
            self.set_params()

        self.x = x
        self.y = y
        return self

    def train(self, n_epoch = 1, lr = 0.1, plot_rate = None, decay = 1.0):
        x = self.x
        y = self.y

        last_p = 0
        for epoch in range(n_epoch):
            c = 0.0
            for i in np.random.permutation(self.N_samples):
                g = grad(lambda p: NN_interpolation.cost(self, x[i], y[i], p))
                dp = g(self.p)
                
                norm = 1.0 / (self.N_samples * self.n_output)

                l = 0
                for dW, db in dp:
                    W, b = self.p[l]
                    W -= lr * dW * norm
                    b -= lr * db * norm
                    self.p[l] = W, b
                    l += 1
                c += self.cost(x[i], y[i])
            c = c / self.N_samples
            lr *= decay

            if plot_rate and epoch//plot_rate >= last_p:
                last_p +=1
                self.plot()

            print(f"Epoch: {epoch}     Cost: {round(float(c), 3)}     LR: {lr}")
        return self

    def plot(self):
        if self.n_input == 1 and self.n_output == 1:
            print("Plotting...")
            x = self.x
            y = self.y
            y_hat = [float(self.eval(xi)) for xi in x]
            plt.plot(x, y)
            plt.plot(x, y_hat)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        else:
            print('Can only plot 1 to 1 functions.')

    def set_weights(self, x, low, up):
        x = 2*(x - 0.5)
        mean = (up + low) * 0.5
        range = (up - low) * 0.5
        return x * range + mean


class NN_classifier:
    def __init__(self, hls = [100], n_input = 1, n_out = 1, lr = 0.1):
        self.set_lrs(hls, n_input, n_out)
        self.n_input = n_input
        self.n_output = n_out
        self.lr = lr

    def set_lrs(self, hls = [100], n_input = 1, n_out = 1):
        self.lrs = [n_input]
        for l in hls:
            self.lrs.append(l)
        self.lrs.append(n_out)
        return self

    def set_init_params(self):
        set_weights = lambda x, low, up: (up - low) * (x - 0.5) + (up + low) * 0.5
        low = np.min(self.x)
        high = np.max(self.x)
        W = set_weights(np.random.rand(self.lrs[1], self.lrs[0]), -1, 1)
        b = set_weights(np.random.rand(self.lrs[1],1), -low, high)
        params = [(W, b)]
        for i in range(1, len(self.lrs) - 1):
            W = set_weights(np.random.rand(self.lrs[i+1], self.lrs[i]), -1, 1)
            b = set_weights(np.random.rand(self.lrs[i+1],1), -1, 1)
            params.append( (W, b) )
        self.p = params
        return self

    def eval(self, x, p = None):
        if self and not p:
            p = self.p

        x = x.reshape(-1, 1)
        # hidden layers
        for W, b in p[:-1]:
            x = np.tanh(np.dot(W, x) + b)
            #print(x)

        # soft max output layer
        W, b = p[-1]
        y = np.exp((np.dot(W, x) + b))
        y = y / np.sum(y)

        return y
    
    def cost(self, x, y, p = None):
        y_hat = NN_classifier.eval(self, x, p)
        sum = 0.0
        for i in range(self.n_output):
            sum -= y[i] * np.log( y_hat[i] )
        return sum

    def set_train_data(self, x, y):
        N_samples_x = len(x) if hasattr(x, '__len__') else 1
        N_samples_y = len(y) if hasattr(y, '__len__') else 1
        if N_samples_x != N_samples_y:
            print("Sample number does not match for x and y.")
            return self
        self.N_samples = N_samples_x
        print(f"There are {N_samples_x} training set samples.")

        N_outputs = len(y[0]) if hasattr(y[0], '__len__') else 1
        n_input = len(x[0]) if hasattr(x[0], '__len__') else 1

        if self.n_input != n_input or self.n_output != N_outputs:
            self.n_input = n_input
            self.n_output = N_outputs
            self.lrs[0] = n_input
            self.lrs[-1] = N_outputs
            print("Updated input and output layers sizes.")

        self.x = x
        self.y = y
        return self

    def get_max_idx(self, x):
        y_hat = self.eval(x)
        return np.where(y_hat == np.amax(y_hat))[0][0]

    def train(self, n_epoch = 1, lr = None, plot_rate = None, decay = 1.0):
        x = self.x
        y = self.y

        if lr:
            self.lr = lr

        norm = 1.0 / self.N_samples

        for epoch in range(n_epoch):
            c = 0
            for i in np.random.permutation(self.N_samples):
                g = grad(lambda p: NN_classifier.cost(self, x[i], y[i], p))
                dp = g(self.p)
                
                l = 0
                for dW, db in dp:
                    W, b = self.p[l]
                    W -= self.lr * dW * norm
                    b -= self.lr * db * norm
                    self.p[l] = W, b
                    l += 1
                if y[i][self.get_max_idx(x[i])] == 1:
                    c += 1
            c = c / self.N_samples
            print(f"Epoch: {epoch}     Accuracy: {round(float(c)*100, 3)}%     LR: {self.lr}")
            self.lr *= decay
        return self

    def save_model(self, file_name):
        # Open a file and use dump()
        with open(file_name + '.pkl', 'wb') as f:
            # A new file will be created
            pickle.dump((self.p, self.lrs, self.lr), f)
        return self

    def load_model(self, file_name):
        # Open the file in binary mode
        with open(file_name + '.pkl', 'rb') as f:
            # Call load method to deserialze
            self.p, self.lrs, self.lr = pickle.load(f)
            self.n_input = self.lrs[0]
            self.n_output = self.lrs[-1]
        return self


def main():
    x = np.linspace(-20, 20, num = 100)
    y = np.sin(x)
    model = NN_interpolation([40, 30, 20, 20, 20], 1, 1)
    model.set_train_data(x, y)
    model.train( 10000, plot_rate = 100, decay = 0.999)
    print(model.eval(3.1416))

def test():
    input_size = 5
    n_outputs = 3
    x = np.random.rand(10, input_size)
    y = np.random.rand(10, n_outputs)
    model = NN_interpolation([40, 30, 20, 20, 20]).set_train_data(x, y)
    model.train(10000, decay = 0.999, plot_rate= 10)

def classifier():
    input_size = 5
    n_outputs = 3
    x = np.random.rand(10, input_size)
    y = np.random.rand(10, n_outputs)
    model = NN_classifier([40, 30, 20, 20, 20]).set_train_data(x, y)
    model.train(10000, decay = 0.999, plot_rate= 10)

if __name__ == '__main__':
    classifier()