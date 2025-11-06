# src/common/audit.py
import json, hashlib, time, os
from typing import Any, Dict

class HashChainedAudit:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._last_hash = "0"*64
        if os.path.exists(path):
            # استرجاع آخر هاش
            with open(path, "rb") as f:
                for line in f:
                    try:
                        rec = json.loads(line.decode("utf-8"))
                        self._last_hash = rec.get("hash", self._last_hash)
                    except Exception:
                        pass

    def append(self, event: Dict[str, Any]):
        event = dict(event)
        event["ts"] = time.time()
        payload = json.dumps(event, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(self._last_hash.encode("utf-8") + payload).hexdigest()
        rec = {"event": event, "prev": self._last_hash, "hash": h}
        with open(self.path, "ab") as f:
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self._last_hash = h
