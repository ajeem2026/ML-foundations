import torch.nn as nn

class SLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)