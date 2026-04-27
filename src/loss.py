"""Supervised contrastive losses ported from Holt et al.'s training code."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """NT-Xent supervised contrastive loss with multiple positives per anchor."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, epitope_labels: torch.Tensor) -> torch.Tensor:
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels_matrix = epitope_labels.unsqueeze(0) == epitope_labels.unsqueeze(1)
        non_self = ~torch.eye(labels_matrix.shape[0], dtype=torch.bool, device=labels_matrix.device)
        positives_mask = labels_matrix & non_self

        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim_matrix) * non_self
        denominator = exp_sim.sum(dim=1).clamp_min(1e-12)

        positives_per_anchor = positives_mask.sum(dim=1)
        valid_anchors = positives_per_anchor > 0
        if not torch.any(valid_anchors):
            return embeddings.sum() * 0.0

        positives = torch.masked_select(exp_sim, positives_mask)
        denominators = denominator.repeat_interleave(positives_per_anchor)
        losses = -torch.log((positives / denominators).clamp_min(1e-12))
        return losses.mean()


class ContrastiveTrainTestLoss(nn.Module):
    """Evaluation-only contrastive losses split into train, test, and cross pairs."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        epitope_labels: torch.Tensor,
        dataset_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
            labels_matrix = epitope_labels.unsqueeze(0) == epitope_labels.unsqueeze(1)
            train_mask = dataset_labels.unsqueeze(0) & dataset_labels.unsqueeze(1)
            test_mask = (~dataset_labels).unsqueeze(0) & (~dataset_labels).unsqueeze(1)
            cross_mask = (dataset_labels.unsqueeze(0) & (~dataset_labels).unsqueeze(1)) | (
                (~dataset_labels).unsqueeze(0) & dataset_labels.unsqueeze(1)
            )

            non_self = ~torch.eye(labels_matrix.shape[0], dtype=torch.bool, device=labels_matrix.device)
            train_mask = train_mask & non_self
            test_mask = test_mask & non_self

            sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()
            exp_sim = torch.exp(sim_matrix)

            def _masked_loss(positive_mask: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
                positives_per_anchor = positive_mask.sum(dim=1)
                if positive_mask.sum() == 0:
                    return torch.tensor(0.0, device=embeddings.device)
                denominator = (exp_sim * sample_mask).sum(dim=1).clamp_min(1e-12)
                positives = torch.masked_select(exp_sim, positive_mask)
                denominators = denominator.repeat_interleave(positives_per_anchor)
                return -torch.log((positives / denominators).clamp_min(1e-12)).mean()

            train_loss = _masked_loss(labels_matrix & train_mask, train_mask)
            test_loss = _masked_loss(labels_matrix & test_mask, test_mask)
            cross_loss = _masked_loss(labels_matrix & cross_mask, cross_mask)
            return train_loss, test_loss, cross_loss
