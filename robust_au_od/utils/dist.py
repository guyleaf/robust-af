from torch.utils.data import get_worker_info


def get_worker_id() -> int:
    worker_info = get_worker_info()
    if worker_info is None:
        return 0
    else:
        return worker_info.id
