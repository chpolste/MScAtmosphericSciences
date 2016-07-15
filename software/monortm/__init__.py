"""Wraps some of MONORTMs functionality (mainly IATM=0 stuff).

Dependencies:
- fortranformat (https://bitbucket.org/brendanarnold/py-fortranformat/)
"""

__all__ = ["paths"]

paths = {
        "MonoRTM": "/home/chris/local/monortm/monortm_v5.2_linux_gnu_dbl",
        "TAPE3": "/home/chris/local/monortm/TAPE3"
        }

from monortm.core import MonoRTM, write
from monortm import records

