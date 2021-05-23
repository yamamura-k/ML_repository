from mytorch import Layer
from mytorch import utils
import mytorch.functions as F
import mytorch.layers as L

class Model(Layer):
    def plot(self, *inputs, save_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, save_file=save_file)
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)