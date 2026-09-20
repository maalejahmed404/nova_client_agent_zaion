import json
import os
import urllib.request
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
)
d = json.load(urllib.request.urlopen(req))["data"]
print(f"usage ${d['usage']:.2f}  limit ${d['limit']}  remaining ${d['limit_remaining']:.2f}")
