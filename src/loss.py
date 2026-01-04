import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        similarity = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))

        target = torch.arange(batch_size, device=z.device)
        target = torch.cat([target + batch_size, target], dim=0)
        return F.cross_entropy(similarity, target)
