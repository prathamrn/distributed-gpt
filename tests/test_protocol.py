"""Protects dgpt/protocol.py, the wire format: round-trip fidelity, the bf16 byte halving"""
import pytest
import torch

from dgpt.model import GPT, GPTConfig
from dgpt.protocol import check_layout, config_hash, fingerprint, layout_of, pack, unpack


@pytest.fixture(scope="module")
def sd():
    """- A real seeded GPT state dict (tied embedding included), built once per module so fingerprints are stable."""
    torch.manual_seed(0)
    return {k: v.clone() for k, v in GPT(GPTConfig()).state_dict().items()}


def test_fp32_roundtrip_exact(sd):
    """- Catches a silent precision change or lost version metadata on the download path."""
    t, meta = unpack(pack(sd, {"version": 7}, "float32"))
    assert meta == {"version": 7}
    for k in sd:
        assert torch.equal(t[k], sd[k])


def test_bf16_roundtrip_within_bf16_precision(sd):
    """- Catches a receiver that leaves tensors in bf16, pushing a wire-only precision into the merge."""
    t, _ = unpack(pack(sd, {}, "bfloat16"))
    for k in sd:
        assert t[k].dtype == torch.float32
        assert torch.allclose(t[k], sd[k], rtol=1e-2, atol=1e-6)


def test_bf16_is_half_the_bytes(sd):
    """- Catches an fp32 fallback that doubles every run's bandwidth (21 MB vs 43 MB) with the loss unchanged."""
    a, b = len(pack(sd, {}, "float32")), len(pack(sd, {}, "bfloat16"))
    assert abs(a / b - 2.0) < 0.05


def test_unpack_rejects_garbage():
    """- A body that is not the wire format, or is truncated, raises instead of yielding tensors."""
    with pytest.raises(Exception):
        unpack(b"\x00" * 4)
    with pytest.raises(Exception):
        unpack(pack({"a": torch.zeros(2)}, {}, "float32")[:-3])   # truncated payload


def test_unpack_never_executes_code():
    """- A pickle-bomb style body must fail to parse, not run anything."""
    import pickle, struct
    evil = pickle.dumps({"__reduce__": "os.system"})
    with pytest.raises(Exception):
        unpack(struct.pack("<Q", len(evil)) + evil)


def test_check_layout(sd):
    """- The coordinator averages tensor by tensor keyed on state_dict names"""
    assert check_layout(sd, layout_of(sd)) is None
    bad = dict(sd); bad["extra"] = torch.zeros(1)
    assert "name mismatch" in check_layout(bad, layout_of(sd))
    bad = dict(sd); k = next(iter(bad)); bad[k] = torch.zeros(3, 3)
    assert "shape mismatch" in check_layout(bad, layout_of(sd))


def test_fingerprint_changes_with_content(sd):
    """- The fingerprint changes when any weight changes and is stable for identical weights."""
    f = fingerprint(sd)
    assert len(f) == 16
    mod = {k: v.clone() for k, v in sd.items()}
    k = next(iter(mod)); mod[k][0] += 1e-3
    assert fingerprint(mod) != f


def test_config_hash_order_independent():
    """- The config hash is what makes every machine agree on one model and recipe"""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
