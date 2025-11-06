# Secure Federated Learning (PyTorch + Flower + Differential Privacy)

> **Ready-to-run project**: realistic FL with PyTorch, Flower, Opacus (DP-SGD) + non-IID partitioning + simple hash-chained audit log. Optional TenSEAL HE demo included.

---

## 📦 Project Structure
```
secure-fl/
├─ README.md
├─ requirements.txt
├─ server.py
├─ client.py
├─ models.py
├─ dataset.py
├─ dp_utils.py
├─ audit_log.py
└─ he_optional/
   └─ he_demo.py
```

---

## 🚀 Quickstart
1) **Install deps**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
















secure-fl/
├─ README.md
├─ requirements.txt
├─ server.py
├─ client.py
├─ models.py
├─ dataset.py
├─ dp_utils.py
├─ audit_log.py
├─ chain/
│  ├─ FLRegistry.sol
│  ├─ FLRegistry.abi.json
│  ├─ chain_utils.py
│  ├─ web3_integration.py
│  ├─ package.json
│  ├─ hardhat.config.js
│  └─ scripts/
│     └─ deploy.js
└─ he_optional/
   └─ he_demo.py

2) **Start the FL server** (terminal 1)
```bash
python server.py --rounds 5 --num_clients 5 --min_available 5
```

3) **Start clients** (open 5 terminals or background processes)
```bash
# Terminal 2..6 (change --client_id for each):
python client.py --client_id 1 --num_clients 5 --local_epochs 2 --batch_size 64 \
  --dp --noise_multiplier 1.2 --max_grad_norm 1.0 --noniid_alpha 0.3
```

4) **(Optional) Run HE demo**
```bash
python he_optional/he_demo.py
```

> Dataset: `sklearn.datasets.load_breast_cancer` (طبي/ثدي) — binary classification. Non‑IID يقسّم البيانات على العملاء باستخدام Dirichlet.

---

## 🧰 requirements.txt
```txt
flwr==1.7.0
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opacus>=1.4.0
joblib>=1.3.0
loguru>=0.7.2
# Optional: HE demo
tenseal>=0.3.14
```

---

## 📖 README.md
```md
# Secure Federated Learning (PyTorch + Flower + Differential Privacy)

This repo spins up a realistic FL simulation: a central server orchestrates multiple PyTorch clients. Each client trains locally with DP-SGD (Opacus), using a non-IID data shard. The server aggregates with FedAvg. We also keep a hash‑chained audit log of key events.

## How to run
See the Quickstart in the main canvas.

### Tuning privacy
- Increase `--noise_multiplier` for stronger DP (higher noise → lower accuracy).
- `--max_grad_norm` controls clipping.
- Epsilon is reported per round (approximate), computed from Opacus’ accountant.

### Non-IID control
- `--noniid_alpha` (Dirichlet). Smaller alpha → more skew → harder FL.

### Notes
- This is production‑leaning but still a local sim. For hospitals: replace the dataset loader with EHR connectors, run clients on-site, and keep DP on clients. Consider secure transport (TLS), auth, and HSM‑backed key mgmt; consider true secure aggregation and HE where appropriate.
```

---

## 🧠 models.py
```python
from typing import Tuple
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def model_and_loss(in_features: int, num_classes: int = 2) -> Tuple[nn.Module, nn.Module]:
    model = MLP(in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    return model, criterion
```

---

## 🧪 dataset.py
```python

```

---

## 🔐 dp_utils.py
```python

```

---

## 🧾 audit_log.py
```python

```

---

## 🖥️ server.py
```python

```

---

## 👩‍⚕️ 
```python

```

---

## 🏛️ he_optional/he_demo.py (CKKS aggregation demo)
```python
"""
Optional: Demonstrate real homomorphic operations (CKKS) with TenSEAL.
- Encrypt float vectors from two "clients"
- Server adds ciphertexts and scales average
- Decrypt to get the averaged vector
Note: This demo is stand-alone (not wired into Flower) to keep the main FL runnable everywhere.
"""
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
```

---

### ✅ What makes this “realistic”
- **Flower orchestration** with proper client/server processes
- **Opacus DP-SGD** on clients (actual ε reported)
- **Non-IID Dirichlet** partitioning
- **Audit log** with hash chaining (append-only)
- **HE demo** using TenSEAL performing real homomorphic ops on vectors

### 🛠️ For hospital pilots
- Run `client.py` on each site (on-prem), connect to a central `server.py`
- Replace `dataset.py` with loaders for EHR tables (after de-identification)
- Keep DP on client; enable TLS for transport; integrate real auth/PKI
- Swap audit log with a permissioned ledger (e.g., Fabric) if needed
- Consider secure aggregation / HE at scale (e.g., use a dedicated trusted aggregator)

