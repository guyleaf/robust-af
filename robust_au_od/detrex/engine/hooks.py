from detectron2.engine import BestCheckpointer as DETECTRON2_BestCheckpointer


class BestCheckpointer(DETECTRON2_BestCheckpointer):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    FILE_FORMAT = "{}_{:07d}"

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
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(
                    self.FILE_FORMAT.format(self._file_prefix, metric_iter),
                    **additional_state,
                )
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(
                self.FILE_FORMAT.format(self._file_prefix, metric_iter),
                **additional_state,
            )
            self._checkpointer.path_manager.rm(
                self.FILE_FORMAT.format(self._file_prefix, self.best_iter)
            )
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
