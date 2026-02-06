import torch
import numpy as np
from sklearn.metrics import average_precision_score


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    Parameters
    ----------
    relevances : np.ndarray
        Relevance scores in ranked order.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        DCG@k score.
    """
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Parameters
    ----------
    relevances : np.ndarray
        Relevance scores in ranked order.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        NDCG@k score in [0, 1].
    """
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_ndcg(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute mean NDCG@k across query groups.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted scores of shape (n_samples,).
    labels : torch.Tensor
        Binary relevance labels of shape (n_samples,).
    group_ids : torch.Tensor
        Query/session group identifiers of shape (n_samples,).
    k : int
        Number of top items to consider per group.

    Returns
    -------
    float
        Mean NDCG@k across all groups.
    """
    predictions = predictions.detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().numpy().flatten()
    group_ids = group_ids.detach().cpu().numpy().flatten()

    unique_groups = np.unique(group_ids)
    ndcg_scores = []

    for group in unique_groups:
        mask = group_ids == group
        group_preds = predictions[mask]
        group_labels = labels[mask]

        if len(group_labels) == 0 or group_labels.sum() == 0:
            continue

        sorted_indices = np.argsort(-group_preds)
        sorted_labels = group_labels[sorted_indices]

        ndcg = ndcg_at_k(sorted_labels, k)
        ndcg_scores.append(ndcg)

    if len(ndcg_scores) == 0:
        return 0.0
    return float(np.mean(ndcg_scores))


def compute_prauc(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Precision-Recall Area Under Curve.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted probabilities.
    labels : torch.Tensor
        Binary labels.

    Returns
    -------
    float
        PR-AUC score.
    """
    predictions = predictions.detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().numpy().flatten()

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0

    return float(average_precision_score(labels, predictions))


class MetricsCalculator:
    """
    Accumulates predictions and computes ranking metrics.

    Designed for batch-wise accumulation during evaluation.
    """

    def __init__(self, ctr_loss_weight: float = 1.0, ctcvr_loss_weight: float = 5.0):
        self.ctr_loss_weight = ctr_loss_weight
        self.ctcvr_loss_weight = ctcvr_loss_weight
        self.reset()

    def reset(self):
        """Reset accumulated predictions and labels."""
        self.ctr_preds = []
        self.ctr_labels = []
        self.ctcvr_preds = []
        self.ctcvr_labels = []
        self.group_ids = []

    def update(
        self,
        p_ctr: torch.Tensor,
        p_ctcvr: torch.Tensor,
        click_labels: torch.Tensor,
        conversion_labels: torch.Tensor,
        group_ids: torch.Tensor,
    ):
        """
        Accumulate batch predictions and labels.

        Parameters
        ----------
        p_ctr : torch.Tensor
            Predicted click probabilities.
        p_ctcvr : torch.Tensor
            Predicted conversion probabilities.
        click_labels : torch.Tensor
            Binary click labels.
        conversion_labels : torch.Tensor
            Binary conversion labels.
        group_ids : torch.Tensor
            Query/session group identifiers.
        """
        self.ctr_preds.append(p_ctr.detach())
        self.ctr_labels.append(click_labels.detach())
        self.ctcvr_preds.append(p_ctcvr.detach())
        self.ctcvr_labels.append(conversion_labels.detach())
        self.group_ids.append(group_ids.detach())

    def compute(self, k_values: list[int] | None = None) -> dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Parameters
        ----------
        k_values : list[int] | None
            List of k values for NDCG computation. Defaults to [10, 38].

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.
        """
        if k_values is None:
            k_values = [10, 38]  # 38 mainly because the expedia benchmark evals on ndcg#38

        ctr_preds = torch.cat(self.ctr_preds)
        ctr_labels = torch.cat(self.ctr_labels)
        ctcvr_preds = torch.cat(self.ctcvr_preds)
        ctcvr_labels = torch.cat(self.ctcvr_labels)
        group_ids = torch.cat(self.group_ids)

        w_labels = ctcvr_labels * self.ctcvr_loss_weight + ctr_labels * (self.ctr_loss_weight - ctcvr_labels)

        metrics = {
            "ctr_prauc": compute_prauc(ctr_preds, ctr_labels),
            "ctcvr_prauc": compute_prauc(ctcvr_preds, ctcvr_labels),
        }

        for k in k_values:
            metrics[f"ctr_ndcg@{k}"] = compute_ndcg(ctr_preds, w_labels, group_ids, k)
            metrics[f"ctcvr_ndcg@{k}"] = compute_ndcg(
                ctcvr_preds, w_labels, group_ids, k
            )

        return metrics