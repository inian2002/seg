def load_model_state(model, state_dict, verbose=True):
    """
    Backward-compatible loading:
    - old checkpoints can miss recon_head.*
    - future checkpoints can contain extra keys we do not need
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))

    recon_missing = [k for k in missing if k.startswith('recon_head.')]
    core_missing = [k for k in missing if not k.startswith('recon_head.')]

    if verbose:
        if recon_missing:
            print("  [ckpt] old checkpoint detected: recon head will be randomly initialized")
        if core_missing:
            print(f"  [ckpt] missing core keys: {core_missing}")
        if unexpected:
            print(f"  [ckpt] unexpected keys ignored: {unexpected}")

    return incompatible
