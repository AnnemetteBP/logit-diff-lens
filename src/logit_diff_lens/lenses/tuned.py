from tuned_lens.nn.lenses import TunedLens


def load_pretrained_tuned_lens(*, model, resource_id: str, map_location=None) -> TunedLens:
    lens = TunedLens.from_model_and_pretrained(model, resource_id, map_location=map_location)
    lens.eval()
    return lens


__all__ = [
    "TunedLens",
    "load_pretrained_tuned_lens",
]
