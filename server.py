import argparse
import os
from typing import List, Tuple, Dict, Any
from loguru import logger
import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch

from audit_log import HashChainedAudit
from models import model_and_loss
from dataset import load_data

# بلوكتشين إلزامي
from chain.chain_utils import state_hash_hex, to_bytes32
from chain.web3_integration import record_round_final

def agg_fit(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    eps_pairs = [(n, m.get("dp_epsilon")) for n, m in metrics if m.get("dp_epsilon", -1) >= 0]
    if eps_pairs:
        total_n = sum(n for n, _ in eps_pairs)
        mean_eps = sum(n * float(e) for n, e in eps_pairs) / max(total_n, 1)
        logger.info("Aggregated dp_epsilon (weighted mean) = {}", mean_eps)
        return {"dp_epsilon_mean": float(mean_eps)}
    return {}

def agg_eval(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    acc_pairs = [(n, m["accuracy"]) for n, m in metrics if "accuracy" in m]
    if acc_pairs:
        total_n = sum(n for n, _ in acc_pairs)
        mean_acc = sum(n * float(a) for n, a in acc_pairs) / max(total_n, 1)
        logger.info("Aggregated accuracy (weighted mean) = {}", mean_acc)
        return {"accuracy_mean": float(mean_acc)}
    return {}

class SaveAndSealFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.final_params = None
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd, results, failures):
        agg, metrics = super().aggregate_fit(rnd, results, failures)
        if agg is not None:
            self.final_params = agg
            # احسب بصمة النموذج العالمي لكل جولة واختمها
            try:
                nds = parameters_to_ndarrays(agg)
                X_train, *_ = load_data()
                in_features = X_train.shape[1]
                model, _ = model_and_loss(in_features)
                state = model.state_dict()
                for (k, _), p in zip(state.items(), nds):
                    state[k] = torch.tensor(p)
                model.load_state_dict(state)
                global_hex = state_hash_hex(model.state_dict())
                record_round_final(
                    round_id=int(rnd),
                    global_hash_b32=to_bytes32(global_hex),
                    model_uri="",  # ضع URI خارجي (IPFS/S3) إن وُجد
                    num_clients=int(self.min_available_clients or 0),
                )
            except Exception as e:
                logger.warning("Chain round seal failed at rnd {}: {}", rnd, e)
        return agg, metrics

def get_strategy(min_available: int, num_rounds: int) -> fl.server.strategy.FedAvg:
    return SaveAndSealFedAvg(
        num_rounds=num_rounds,
        min_fit_clients=min_available,
        min_evaluate_clients=min_available,
        min_available_clients=min_available,
        on_fit_config_fn=lambda rnd: {"round": rnd, "num_rounds": num_rounds},
        on_evaluate_config_fn=lambda rnd: {"round": rnd, "num_rounds": num_rounds},
        fit_metrics_aggregation_fn=agg_fit,
        evaluate_metrics_aggregation_fn=agg_eval,
    )

def main():
    parser = argparse.ArgumentParser(description="Secure FL Server (Flower + Mandatory Blockchain)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--min_available", type=int, default=5)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--round_timeout", type=float, default=0.0, help="0=wait forever; else seconds")
    args = parser.parse_args()

    audit = HashChainedAudit("audit_log.jsonl")
    audit.append({"type": "server_start", "rounds": args.rounds, "min_available": args.min_available, "addr": args.server_address})

    strategy = get_strategy(args.min_available, args.rounds)

    logger.info("Starting Flower server at {} (rounds={})", args.server_address, args.rounds)
    fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(
            num_rounds=args.rounds,
            round_timeout=args.round_timeout if args.round_timeout > 0 else None,
        ),
    )

    # احفظ الموديل العالمي النهائي
    if strategy.final_params is not None:
        nds = parameters_to_ndarrays(strategy.final_params)
        X_train, *_ = load_data()
        in_features = X_train.shape[1]
        model, _ = model_and_loss(in_features)
        state = model.state_dict()
        for (k, _), p in zip(state.items(), nds):
            state[k] = torch.tensor(p)
        model.load_state_dict(state)

        os.makedirs("artifacts", exist_ok=True)
        out_path = f"artifacts/global_model_round_{args.rounds}.pth"
        torch.save(model.state_dict(), out_path)
        audit.append({"type": "server_saved_model", "path": out_path})
        logger.info("✅ Saved final global model to {}", out_path)

        # ختم نهائي للجولة الأخيرة (مرة أخرى لأثر واضح)
        try:
            global_hex = state_hash_hex(model.state_dict())
            record_round_final(
                round_id=int(args.rounds),
                global_hash_b32=to_bytes32(global_hex),
                model_uri="",  # URI خارجي إن أردت
                num_clients=int(args.num_clients),
            )
            audit.append({"type": "chain_round_final", "round": args.rounds, "hash": global_hex})
        except Exception as e:
            audit.append({"type": "chain_error_round_final", "err": str(e)})
            logger.warning("Final chain seal failed: {}", e)

    audit.append({"type": "server_stop"})
    logger.info("Flower server stopped.")

if __name__ == "__main__":
    # تحقّق صارم: متغيرات البيئة مطلوبة قبل تشغيل الخادم
    for var in ["WEB3_RPC", "WEB3_PRIVATE_KEY", "FL_CONTRACT"]:
        if not os.getenv(var):
            raise RuntimeError(f"[CHAIN] Missing required env var: {var}")
    main()
