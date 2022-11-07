import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module): # inspired by : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.06, device="cuda:0"): # after hyperparam optimization, temperature of 0.06 gave the best ICBHI score
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, projection1, projection2, labels=None):

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss

class SupConCELoss(nn.Module):
    def __init__(self, weights, alpha=0.5, device="cuda:0", temperature=0.06): # after hyperparam optimization, loss tradeoff of 0.5 gave the best score
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature, device=device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.alpha = alpha

    def forward(self, projection1, projection2, prediction1, prediction2, target):

        predictions = torch.cat([prediction1, prediction2], dim=0)
        labels = torch.cat([target, target], dim=0)
        return self.alpha * self.supcon(projection1, projection2, target) + (1 - self.alpha) * self.ce(predictions, labels)