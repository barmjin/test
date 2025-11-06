import hashlib
import torch

def state_hash_hex(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        v = state_dict[k].detach().cpu().contiguous()
        h.update(v.numpy().tobytes())
    return h.hexdigest()

def to_bytes32(hex_str: str) -> bytes:
    b = bytes.fromhex(hex_str)
    if len(b) != 32:
        raise ValueError("Hash must be 32 bytes")
    return b
