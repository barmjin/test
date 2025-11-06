import numpy as np
import tenseal as ts

# Create CKKS context (public params), generate secret key
poly_mod_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()

# Secret key stays with the trusted party (here, same process for demo)
# In production: only share context with public keys to clients and aggregator

# Simulate two clients producing float vectors (e.g., model deltas)
v1 = np.random.randn(1024).astype(np.float64)
v2 = np.random.randn(1024).astype(np.float64)

# Clients encrypt with public context
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

# Aggregator adds ciphertexts and computes average without decrypting
enc_sum = enc_v1 + enc_v2
enc_avg = enc_sum * (1.0/2.0)

# Trusted party decrypts aggregate
avg = np.array(enc_avg.decrypt())
# Check closeness
print("MSE to plaintext avg:", np.mean((avg - (v1+v2)/2.0)**2))
print("OK: homomorphic sum+scale works on encrypted vectors")