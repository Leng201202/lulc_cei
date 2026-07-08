"""Helpers for loading model weights from a checkpoint file.

Supports two on-disk formats:

* this project's own checkpoints, a dict with a ``model_state_dict`` key
  (see ``train.save_checkpoint``);
* a bare ``state_dict`` (an ``OrderedDict`` of tensors), which is how the
  external OpenEarthMap-SAR baseline weights are distributed.
"""


def extract_state_dict(checkpoint):
    """Return the parameter ``state_dict`` from a loaded checkpoint object."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model_weights(model, checkpoint):
    """Load ``checkpoint`` weights into ``model`` in place.

    Handles both key layouts a wrapped model (e.g. ``LeadingChannelDrop``) can
    encounter:

    * a checkpoint saved from the wrapper itself has ``inner.``-prefixed keys
      and loads straight into ``model``;
    * an external checkpoint (e.g. the OpenEarthMap-SAR baseline) has unprefixed
      keys that belong to the wrapped network, exposed as ``model.inner``.

    We try the wrapper first and fall back to the inner module on a key
    mismatch, so plain models and both wrapped cases all work.
    """
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
        return model
    except RuntimeError:
        inner = getattr(model, "inner", None)
        if inner is None:
            raise
        inner.load_state_dict(state_dict, strict=True)
        return model
