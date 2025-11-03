from ultralytics.utils.callbacks import wb

from ...models.train import YOLOv10DetectionTrainer


def on_pretrain_routine_start(trainer: YOLOv10DetectionTrainer):
    """Initiate and start project if module is present."""
    wb.on_pretrain_routine_start(trainer)
    wb.wb.run.define_metric("grad_norm/batch", step_metric="grad_norm/step")


def on_fit_epoch_end(trainer: YOLOv10DetectionTrainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.on_fit_epoch_end(trainer)


def on_train_epoch_end(trainer: YOLOv10DetectionTrainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.on_train_epoch_end(trainer)
    wb.wb.run.log(trainer.grad_norm, step=trainer.epoch + 1)


def on_train_end(trainer: YOLOv10DetectionTrainer):
    """Save the best model as an artifact at end of training."""
    wb.on_train_end(trainer)


# NOTE: wandb only supports monotonic increasing. So, I don't use it.
# def optimizer_step(trainer: YOLOv10DetectionTrainer):
#     data = {**trainer.grad_norm, "grad_norm/step": trainer.iter}
#     wb.wb.run.log(data, step=trainer.epoch + 1)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
        # "optimizer_step": optimizer_step,
    }
    if wb.wb is not None
    else {}
)
