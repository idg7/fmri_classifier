from torch.cuda import is_available as gpu_available

import consts


def should_use_cuda() -> bool:
    if gpu_available():
        if not consts.DEBUG:
            return True
    return False
