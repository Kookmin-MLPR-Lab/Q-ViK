
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class VisualSinkDetection:
    mask: torch.Tensor
    scores: torch.Tensor
    layer_index: int
    threshold: float

    @property
    def count(self) -> int:
        return int(self.mask.sum().item())

    @property
    def ratio(self) -> float:
        if self.mask.numel() == 0:
            return 0.0
        return self.count / float(self.mask.numel())


def resolve_layer_index(n_layers: int, requested: int) -> int:
    if n_layers <= 0:
        return 0
    if requested < 0:
        return max(0, n_layers + requested)
    return min(requested, n_layers - 1)


def detect_visual_sinks(
    hidden_states: tuple[torch.Tensor, ...],
    image_positions: torch.Tensor,
    requested_layer: int,
    threshold: float,
    max_ratio: float,
) -> VisualSinkDetection:
    layer_index = resolve_layer_index(len(hidden_states), requested_layer)
    image_hidden = hidden_states[layer_index][0, image_positions.to(hidden_states[layer_index].device), :]
    massive_scores = image_hidden.detach().float().abs().amax(dim=-1).cpu()
    baseline = massive_scores.median().clamp_min(torch.finfo(massive_scores.dtype).eps)
    scores = massive_scores / baseline
    mask = scores >= float(threshold)
    if 0.0 < max_ratio < 1.0:
        max_count = max(1, int(round(float(mask.numel()) * max_ratio)))
        if int(mask.sum().item()) > max_count:
            selected = torch.topk(scores, k=max_count, largest=True).indices
            capped = torch.zeros_like(mask)
            capped[selected] = True
            mask = capped
    return VisualSinkDetection(mask=mask, scores=scores, layer_index=layer_index, threshold=float(threshold))


def topk_mask_with_optional_sink_filter(
    scores: torch.Tensor,
    n_keep: int,
    visual_sink_mask: torch.Tensor,
    filter_visual_sinks: bool,
) -> torch.Tensor:
    if n_keep >= int(scores.numel()):
        return torch.ones(int(scores.numel()), dtype=torch.bool)
    rank_scores = scores.detach().cpu().float().clone()
    if filter_visual_sinks and visual_sink_mask.numel() == rank_scores.numel():
        non_sink_count = int((~visual_sink_mask).sum().item())
        if non_sink_count >= n_keep:
            rank_scores[visual_sink_mask] = -torch.inf
    top = torch.topk(rank_scores, k=n_keep, largest=True).indices
    keep = torch.zeros(int(scores.numel()), dtype=torch.bool)
    keep[top] = True
    return keep
