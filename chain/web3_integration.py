import json, os
from web3 import Web3

# === بيئة إلزامية ===
RPC = os.getenv("WEB3_RPC")
PRIVATE_KEY = os.getenv("WEB3_PRIVATE_KEY")
CONTRACT_ADDR = os.getenv("FL_CONTRACT")

if not RPC or not PRIVATE_KEY or not CONTRACT_ADDR:
    raise RuntimeError("WEB3_RPC / WEB3_PRIVATE_KEY / FL_CONTRACT must be set")

# اتصال بالعقدة
w3 = Web3(Web3.HTTPProvider(RPC))
if not w3.is_connected():
    raise RuntimeError(f"Cannot connect to RPC at {RPC}")

acct = w3.eth.account.from_key(PRIVATE_KEY)

# تحميل ABI والعقد
abi_path = os.path.join(os.path.dirname(__file__), "FLRegistry.abi.json")
with open(abi_path) as f:
    ABI = json.load(f)

CONTRACT = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDR),
    abi=ABI,
)

def _send(tx: dict) -> str:
    """يبني المعاملة، يوقّع، يرسل، وينتظر الإيصال. يُرجع tx hash hex."""
    # أساسيات المعاملة
    tx.setdefault("from", acct.address)
    tx.setdefault("nonce", w3.eth.get_transaction_count(acct.address))
    tx.setdefault("gas", 500_000)

    # EIP-1559 (Hardhat يدعمها)
    tx.setdefault("maxFeePerGas", w3.to_wei(2, "gwei"))
    tx.setdefault("maxPriorityFeePerGas", w3.to_wei(1, "gwei"))
    tx.setdefault("chainId", w3.eth.chain_id)

    # توقيع
    signed = acct.sign_transaction(tx)
    # دعم v5/v6: rawTransaction أو raw_transaction
    raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
    if raw is None:
        raise RuntimeError("SignedTransaction missing raw bytes (web3 version mismatch?)")

    # إرسال + انتظار الإيصال
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.transactionHash.hex()

def record_client_commit(round_id: int, client_id: int, update_hash_b32: bytes, num_samples: int, epsilon: float) -> str:
    eps_i96 = int(epsilon * 1_000_000) if epsilon >= 0 else 0
    tx = CONTRACT.functions.recordClientCommit(
        int(round_id), int(client_id), update_hash_b32, int(num_samples), eps_i96
    ).build_transaction({})
    return _send(tx)

def record_round_final(round_id: int, global_hash_b32: bytes, model_uri: str, num_clients: int) -> str:
    tx = CONTRACT.functions.recordRoundFinal(
        int(round_id), global_hash_b32, model_uri, int(num_clients)
    ).build_transaction({})
    return _send(tx)
