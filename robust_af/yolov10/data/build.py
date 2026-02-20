from ultralytics.utils import colorstr

from .dataset import RobustYOLODataset, YOLODataset


def build_yolo_dataset(
    cfg, img_path, batch, data, mode="train", rect=False, stride=32, robust=False
):
    """Build YOLO Dataset."""
    kwargs = dict(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
    if robust:
        return RobustYOLODataset(**kwargs)
    else:
        return YOLODataset(**kwargs)
