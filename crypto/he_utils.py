# crypto/he_utils.py
import base64
import numpy as np
import tenseal as ts
from typing import List, Tuple

def make_ckks_context(poly_mod_degree: int = 8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale: float = 2**40) -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, list(coeff_mod_bit_sizes))
    ctx.generate_galois_keys()
    ctx.global_scale = global_scale
    ctx.make_context_public()  # نحتاج مشاركة public + galois فقط؛ السرّ يبقى لدى من يفك
    return ctx

def serialize_public_context(ctx: ts.Context) -> bytes:
    return ctx.serialize(save_public=True, save_secret_key=False)

def encrypt_vector(vec: np.ndarray, ctx: ts.Context) -> bytes:
    enc = ts.ckks_vector(ctx, vec.astype(np.float64))
    return enc.serialize()

def decrypt_vector(blob: bytes, ctx_secret: ts.Context) -> np.ndarray:
    enc = ts.ckks_vector_from(ctx_secret, blob)
    return np.array(enc.decrypt())

def add_encrypted(blobs: List[bytes], ctx_public: ts.Context) -> bytes:
    acc = None
    for b in blobs:
        v = ts.ckks_vector_from(ctx_public, b)
        if acc is None:
            acc = v
        else:
            acc += v
    return acc.serialize()

def avg_encrypted(blobs: List[bytes], ctx_public: ts.Context, denom: int) -> bytes:
    s = add_encrypted(blobs, ctx_public)
    v = ts.ckks_vector_from(ctx_public, s)
    v /= float(denom)
    return v.serialize()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def b64d(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))
