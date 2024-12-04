import torch
from torch.fx import symbolic_trace
from torchviz import make_dot, make_dot_from_trace

from models.cnns.cnn_classifier import CNNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNClassifier(
    in_channels=3, conv_channel_dims=[64, 128, 256], conv_kernel_dims=[3, 3, 3]
).to(device)

x = torch.randn(8, 3, 32, 32).to(device)

output = model(x)

print(output)

# traced_model = symbolic_trace(model)
# dot = make_dot(output, params={"x": x})
# dot.render("cnn_graph", format="png")

# dot.view()
