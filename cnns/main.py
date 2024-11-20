from CNN import CNN
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN(
    in_channels=3,
    conv_channel_dims=[64, 128, 256],
    conv_kernel_dims=[3, 3, 3]
).to(device)

x = torch.randn(8, 3, 32, 32).to(device)

output = model(x)
print(output)