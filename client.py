import argparse
from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from loguru import logger
import flwr as fl

from models import model_and_loss
from dataset import load_data, dirichlet_noniid_split
from dp_utils import attach_privacy, get_epsilon
from audit_log import HashChainedAudit

# بلوكتشين إلزامي: فشل صريح إذا لم تتوفر البيئة/العقد
from chain.chain_utils import state_hash_hex, to_bytes32
from chain.web3_integration import record_client_commit

def get_data_for_client(client_id: int, num_clients: int, batch_size: int, noniid_alpha: float) -> Tuple[DataLoader, DataLoader, int]:
    X_train, y_train, X_test, y_test = load_data()
    idx_map = dirichlet_noniid_split(X_train, y_train, num_clients=num_clients, alpha=noniid_alpha)
    cid0 = (client_id - 1) % num_clients  # دعم IDs من 1..N
    idx = idx_map[cid0]
    Xc, yc = X_train[idx], y_train[idx]
    train_ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(yc))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    in_features = X_train.shape[1]
    return train_loader, test_loader, in_features

def get_params(model: torch.nn.Module):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.Module, params):
    state_dict = model.state_dict()
    for (k, _), p in zip(state_dict.items(), params):
        state_dict[k] = torch.tensor(p)
    model.load_state_dict(state_dict)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_clients: int, local_epochs: int, batch_size: int,
                 lr: float, dp: bool, noise_multiplier: float, max_grad_norm: float, noniid_alpha: float):
        self.cid = cid
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dp = dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.noniid_alpha = noniid_alpha

        self.train_loader, self.test_loader, in_features = get_data_for_client(
            client_id=cid, num_clients=num_clients, batch_size=batch_size, noniid_alpha=noniid_alpha
        )
        self.model, self.criterion = model_and_loss(in_features)
        self.audit = HashChainedAudit(f"audit_client_{cid}.jsonl")
        self.audit.append({"type": "client_start", "cid": cid})

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        round_id = int(config.get("round", 0))
        set_params(self.model, parameters)
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        pe = None
        sample_rate = None
        if self.dp:
            pe, sample_rate, self.model, optimizer, self.train_loader = attach_privacy(
                self.model,
                optimizer,
                self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                # secure_mode=True,  # للإنتاج
            )

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                optimizer.step()

        eps = None
        if self.dp and pe is not None:
            try:
                eps = get_epsilon(pe, sample_rate, epochs=self.local_epochs)
            except Exception:
                eps = None

        # إلزامي: ختم بصمة تحديث العميل على السلسلة
        update_hex = state_hash_hex(self.model.state_dict())
        record_client_commit(
            round_id=round_id,
            client_id=int(self.cid),
            update_hash_b32=to_bytes32(update_hex),
            num_samples=len(self.train_loader.dataset),
            epsilon=float(eps or 0.0),
        )
        self.audit.append({"type": "chain_client_commit", "round": round_id, "hash": update_hex})

        metrics = {"dp_epsilon": float(eps) if eps is not None else -1.0, "round": round_id}
        self.audit.append({"type": "fit_done", "cid": self.cid, "round": round_id, "dp_eps": float(metrics["dp_epsilon"])})
        return get_params(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss_sum += float(loss.detach().cpu().item()) * yb.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum())
                total += int(yb.size(0))
        acc = correct / max(total, 1)
        self.audit.append({"type": "eval", "cid": self.cid, "acc": acc})
        return float(loss_sum / max(total, 1)), total, {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise_multiplier", type=float, default=1.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noniid_alpha", type=float, default=0.3)
    args = parser.parse_args()

    logger.info(
        "Client {} starting (epochs={}, batch={}, dp={}, noise={}, C={} clients)",
        args.client_id, args.local_epochs, args.batch_size, args.dp, args.noise_multiplier, args.num_clients
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            cid=args.client_id,
            num_clients=args.num_clients,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dp=args.dp,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            noniid_alpha=args.noniid_alpha,
        ).to_client(),
    )

if __name__ == "__main__":
    # تحقّق صارم: متغيرات البيئة مطلوبة قبل بدء العميل
    for var in ["WEB3_RPC", "WEB3_PRIVATE_KEY", "FL_CONTRACT"]:
        if not os.getenv(var):
            raise RuntimeError(f"[CHAIN] Missing required env var: {var}")
    main()
