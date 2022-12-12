import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNet

class LogisticRegression(nn.Module):
    """Model for PyTorch logistic regression."""

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # one output for binary classification
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class ThreeLayerNet(nn.Module):
    """Three-layer MLP for classification."""

    def __init__(self, input_size, h1_size=None, dropout=0.5):
        super(ThreeLayerNet, self).__init__()
        # three layers of decreasing size
        if h1_size is None:
            h1_size = input_size // 2
        self.fc0 = nn.Linear(input_size, h1_size)
        self.fc1 = nn.Linear(h1_size, h1_size // 2)
        self.fc2 = nn.Linear(h1_size // 2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class LinearCSD(nn.Module):
    """Model for logistic regression with CSD loss."""

    def __init__(self, input_size, num_domains, k=2):
        super(LinearCSD, self).__init__()
        self.num_domains = num_domains
        self.k = k

        # latent variables, one common and k specific
        self.sms = nn.Parameter(
            torch.normal(0, 1e-1, size=[k+1, input_size, 2],
                         dtype=torch.float, device='cuda'),
            requires_grad=True
        )
        # biases for LVs, one common and k specific
        self.sm_biases = nn.Parameter(
            torch.normal(0, 1e-1, size=[k+1, 2],
                         dtype=torch.float, device='cuda'),
            requires_grad=True
        )

        # weights for LVs, one weight vector for each domain
        self.betas = nn.Parameter(
            torch.normal(0, 1e-4, size=[num_domains, k],
                         dtype=torch.float, device='cuda'),
            requires_grad=True
        )
        # scalar weight for common vs. specific loss
        self.cs_wt = nn.Parameter(
            torch.normal(.1, 1e-4, size=[],
                         dtype=torch.float, device='cuda'),
            requires_grad=True
        )

    def _csd(self,
             embeds,
             domains,
             num_classes,
             num_domains,
             K=1):
        """CSD layer to be used as a replacement for final classification layer

        Adapted from: https://gist.github.com/vihari/0dc2c296e74636725cfee364637fb4f7

        Args:
          embeds (tensor): final layer representations of dim 2
          domains (tensor): tf tensor with domain index of dim 1 -- set to all zeros when testing
          num_classes (int): Number of label classes: scalar
          num_domains (int): Number of domains: scalar
          K (int): Number of domain specific components to use. should be >=1 and <=num_domains-1
        Returns:
          tuple of final loss, logits
        """
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(embeds, w_c) + b_c

        domains = F.one_hot(domains, num_domains).type(torch.FloatTensor).cuda()
        c_wts = torch.matmul(domains, self.betas)

        batch_size = embeds.shape[0]
        c_wts = torch.cat((
            torch.ones((batch_size, 1), dtype=torch.float, device='cuda') * self.cs_wt,
            c_wts
        ), 1)
        c_wts = torch.tanh(c_wts)
        w_d = torch.einsum("bk,krl->brl", c_wts, self.sms)
        b_d = torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, embeds) + b_d

        sms = self.sms
        diag_tensor = torch.stack(
            [torch.eye(K+1) for _ in range(num_classes)],
            dim=0
        ).cuda()
        cps = torch.stack(
            [torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1))
             for _ in range(num_classes)],
            dim=0
        )
        orth_loss = torch.mean((cps - diag_tensor)**2)

        return logits_specialized, logits_common, orth_loss

    def forward(self, x):
        """Forward pass function

        NOTE this function expects the first column of x to be the sample
        domain information. Here we'll separate the domain info from the
        other features, and pass it to the CSD layer to calculate the loss
        components.
        """
        return self._csd(x[:, 1:],
                         x[:, 0].flatten().type(torch.long),
                         2,
                         self.num_domains,
                         self.k)


class CSDClassifier(NeuralNet):

    def __init__(self, *args, **kwargs):
        super(CSDClassifier, self).__init__(*args, **kwargs)
        self.classes_ = [0, 1]

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # override loss function to compute/minimize CSD loss
        logits_specialized, logits_common, orth_loss = y_pred
        criterion = nn.CrossEntropyLoss()
        specific_loss = criterion(logits_specialized, y_true.cuda())
        class_loss = criterion(logits_common, y_true.cuda())
        return class_loss + specific_loss + orth_loss

    def predict_proba(self, X, domains=None):
        # override predict_proba to use only common logits for prediction
        return self.module_.forward(
            torch.from_numpy(X).cuda()
        )[1].cpu().detach().numpy()

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

