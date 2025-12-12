import torch
from detectron2.utils.events import (
    CommonMetricPrinter as ORIGINAL_CommonMetricPrinter,
)
from detectron2.utils.events import get_event_storage


class CommonMetricPrinter(ORIGINAL_CommonMetricPrinter):
    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            avg_data_time = storage.history("data_time").avg(
                storage.count_samples("data_time", self._window_size)
            )
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            avg_data_time = None
            last_data_time = None
        try:
            avg_iter_time = storage.history("time").global_avg()
            last_iter_time = storage.history("time").latest()
        except KeyError:
            avg_iter_time = None
            last_iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        try:
            grad_norm = "{:.5g}".format(storage.history("grad_norm").latest())
            if grad_norm == -1:
                grad_norm = "N/A"
        except KeyError:
            grad_norm = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            str.format(
                " {eta}iter: {iter}  {losses}  {non_losses}  {avg_time}{last_time}"
                + "{avg_data_time}{last_data_time} lr: {lr} grad_norm: {grad_norm}  {memory}",
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(
                            k, v.median(storage.count_samples(k, self._window_size))
                        )
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                non_losses="  ".join(
                    [
                        "{}: {:.4g}".format(
                            k, v.median(storage.count_samples(k, self._window_size))
                        )
                        for k, v in storage.histories().items()
                        if "[metric]" in k
                    ]
                ),
                avg_time=(
                    "time: {:.4f}  ".format(avg_iter_time)
                    if avg_iter_time is not None
                    else ""
                ),
                last_time=(
                    "last_time: {:.4f}  ".format(last_iter_time)
                    if last_iter_time is not None
                    else ""
                ),
                avg_data_time=(
                    "data_time: {:.4f}  ".format(avg_data_time)
                    if avg_data_time is not None
                    else ""
                ),
                last_data_time=(
                    "last_data_time: {:.4f}  ".format(last_data_time)
                    if last_data_time is not None
                    else ""
                ),
                lr=lr,
                grad_norm=grad_norm,
                memory="max_mem: {:.0f}M".format(max_mem_mb)
                if max_mem_mb is not None
                else "",
            )
        )
