from __future__ import annotations

from typing import Literal

import torch as th
from transformers import PreTrainedModel

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens
from tuned_lens.nn.unembed import Unembed


ReadoutKind = Literal["raw", "model_norm", "tuned"]


class RawUnembedLens(Lens):
    """Local raw readout adapter that matches the tuned-lens Lens interface."""

    def __init__(self, model: PreTrainedModel):
        super().__init__(Unembed(model))

    @classmethod
    def from_model(cls, model: PreTrainedModel) -> "RawUnembedLens":
        return cls(model)

    def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
        del idx
        return h

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        del idx
        return self.unembed.unembedding(h)


class ModelNormUnembedLens(Lens):
    """Local ModelNorm adapter using the model's own final readout normalization."""

    def __init__(self, model: PreTrainedModel):
        super().__init__(Unembed(model))

    @classmethod
    def from_model(cls, model: PreTrainedModel) -> "ModelNormUnembedLens":
        return cls(model)

    def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
        del idx
        return self.unembed.final_norm(h)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        del idx
        return self.unembed.forward(h)


def build_tuned_lens_adapter(
    *,
    model: PreTrainedModel,
    kind: ReadoutKind,
    tuned_lens_resource: str | None = None,
    map_location: str | th.device | None = None,
) -> Lens:
    """Construct a local lens object compatible with tuned-lens plotting utilities."""

    if kind == "raw":
        lens = RawUnembedLens.from_model(model)
    elif kind == "model_norm":
        lens = ModelNormUnembedLens.from_model(model)
    elif kind == "tuned":
        if not tuned_lens_resource:
            raise ValueError("tuned_lens_resource is required when kind='tuned'.")
        lens = TunedLens.from_model_and_pretrained(
            model,
            lens_resource_id=tuned_lens_resource,
            map_location=map_location,
        )
    else:
        raise ValueError(f"Unsupported lens kind: {kind!r}")

    lens.eval()
    return lens


__all__ = [
    "LogitLens",
    "Lens",
    "ModelNormUnembedLens",
    "RawUnembedLens",
    "ReadoutKind",
    "TunedLens",
    "build_tuned_lens_adapter",
]
