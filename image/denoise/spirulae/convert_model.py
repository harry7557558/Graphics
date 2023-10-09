import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass

def convert_model(path):
    model = torch.load(path)
    for param in model.parameters():
        print(param.shape)

if __name__ == "__main__":
    convert_model("spirulae/seq_implicit3.pth")

