import os
import sys
from pathlib import Path

cur_root = Path(__file__).absolute().parent

os.chdir(cur_root)

sys.path.append(str(cur_root.parent))
