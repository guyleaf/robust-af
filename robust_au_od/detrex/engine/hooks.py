from detectron2.engine import BestCheckpointer as DETECTRON2_BestCheckpointer
from fvcore.common.checkpoint import Checkpointer


class BestCheckpointer(DETECTRON2_BestCheckpointer):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    FILE_FORMAT = "{}_{:07d}"

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
    ):
        super().__init__(
            eval_period, checkpointer, val_metric, mode=mode, file_prefix=file_prefix
        )
        self.path_manager = checkpointer.path_manager
        self.last_checkpoint = None

    def _save_best_checkpoint(self, metric_iter: int):
        additional_state = {"iteration": metric_iter}
        self._checkpointer.save(
            self.FILE_FORMAT.format(self._file_prefix, metric_iter), **additional_state
        )
        # remove previous best checkpoint
        if self.last_checkpoint is not None and self.path_manager.exists(
            self.last_checkpoint
        ):
            self.path_manager.rm(self.last_checkpoint)
        self.last_checkpoint = self._checkpointer.get_checkpoint_file()

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                f"Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                self._save_best_checkpoint(metric_iter)
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):
            self._save_best_checkpoint(metric_iter)
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is "
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
