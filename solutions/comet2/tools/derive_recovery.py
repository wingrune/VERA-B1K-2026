import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from solutions.comet2.recovery import derive_rules
p=Path(__file__).resolve().parents[1]/'assets'
source=p/'gripper_table.source.json'
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'min_closed_fraction':0.002,
        'first_closure':derive_rules(json.loads(source.read_text()))}
assert result==json.loads((p/'recovery_rules.json').read_text()),'Runtime table is not reproducible'
print('All 100 recovery rules reproduce exactly')
