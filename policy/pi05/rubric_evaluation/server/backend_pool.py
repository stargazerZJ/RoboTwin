from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Backend:
    host: str
    port: int
    gpu_id: int


def parse_gpus(gpus: str) -> List[int]:
    gpus = gpus.strip()
    if gpus == "all":
        # Prefer nvidia-smi if available; fallback to CUDA_VISIBLE_DEVICES; else assume 0.
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
            ids = []
            for line in out.splitlines():
                m = re.match(r"GPU\s+(\d+):", line.strip())
                if m:
                    ids.append(int(m.group(1)))
            if ids:
                return ids
        except Exception:
            pass

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            # If user set CUDA_VISIBLE_DEVICES=0,1,2 then treat those as logical ids.
            parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
            try:
                return [int(p) for p in parts]
            except Exception:
                return [0]
        return [0]

    parts = [p.strip() for p in gpus.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def make_backends(host: str, base_port: int, gpu_ids: Sequence[int]) -> List[Backend]:
    return [Backend(host=host, port=base_port + i, gpu_id=int(i)) for i in gpu_ids]