# Types
from typing import Union, Optional

import numpy as np
import torch

ArrayOrTensor = Union[np.ndarray, torch.Tensor]
OptionalArrayOrTensor = Optional[ArrayOrTensor]

# models
from .pairwise import Pairwise
from .pairwise import Pairwise as Ising
from .independent import Independent
from .rp import RP
from .rp import RP as MERP
from .sigmoid_rp import SigmoidRP
from .sigmoid_rp import SigmoidRP as SigmoidMERP
