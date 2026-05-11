# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from source.models import equal_weight

# Test model
w = equal_weight(4)
print('EW weights:', w)
print('Test passed!')