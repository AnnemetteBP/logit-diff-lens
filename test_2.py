import os
from pathlib import Path
path = Path("/media/am/AM/logit-diff-lens/ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice")

print("Exists:", os.path.exists(path))
print("Files:", os.listdir(path))