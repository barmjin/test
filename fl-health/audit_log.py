import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

class HashChainedAudit:
    def __init__(self, filepath: str = "audit_log.jsonl"):
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = "0" * 64
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    last = json.loads(line).get("entry_hash", last)
                except Exception:
                    pass
        return last



    def append(self, event: Dict[str, Any]):
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "prev_hash": self.prev_hash,
        }
        h = hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()
        entry["entry_hash"] = h
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.prev_hash = h
