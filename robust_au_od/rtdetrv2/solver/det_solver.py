import datetime
import json
import time

import torch
from rtdetrv2.core import YAMLConfig
from rtdetrv2.misc import dist_utils
from rtdetrv2.solver import DetSolver as ORIGINAL_DetSolver
from rtdetrv2.solver.det_engine import evaluate, train_one_epoch


class DetSolver(ORIGINAL_DetSolver):
    def __init__(self, cfg: YAMLConfig) -> None:
        super().__init__(cfg)
        assert isinstance(cfg, YAMLConfig), "Only support YAMLConfig instance."
        self.cfg = cfg

    @property
    def with_train_evaluation(self):
        return (
            self.val_train_dataloader is not None and self.train_evaluator is not None
        )

    def train(self):
        super().train()

        # prepare validation version of train_dataloader and train_evaluator
        val_train_dataloader_cfg = self.cfg.yaml_cfg.get("val_train_dataloader")
        if val_train_dataloader_cfg is not None:
            # TODO: hacky way, make an API in RT-DETR repo
            # store originals
            val_dataloader_cfg = self.cfg.yaml_cfg["val_dataloader"]
            val_dataloader = self.cfg._val_dataloader
            evaluator = self.cfg._evaluator

            # reset val_dataloader, evaluator and replace with val_train_dataloader
            self.cfg._evaluator = None
            self.cfg._val_dataloader = None
            self.cfg.yaml_cfg["val_dataloader"] = val_train_dataloader_cfg

            # build dataloader by YAMLConfig API
            val_train_dataloader = self.cfg.val_dataloader
            self.val_train_dataloader = dist_utils.warp_loader(
                val_train_dataloader, shuffle=val_train_dataloader.shuffle
            )

            # build evaluator by YAMLConfig API
            self.train_evaluator = self.cfg.evaluator

            # restore originals
            self.cfg._evaluator = evaluator
            self.cfg._val_dataloader = val_dataloader
            self.cfg.yaml_cfg["val_dataloader"] = val_dataloader_cfg
        else:
            self.val_train_dataloader = self.train_evaluator = None

    def fit(self):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print(f"number of trainable parameters: {n_parameters}")

        best_stat = {
            "epoch": -1,
        }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        for epoch in range(start_epcoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            print("Test evaluation")
            test_stats, coco_evaluator = evaluate(  # noqa: F821
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
            )
            if self.with_train_evaluation:
                print("Train evaluation")
                test_train_stats, _ = evaluate(  # noqa: F821
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_train_dataloader,
                    self.train_evaluator,
                    self.device,
                )
            else:
                test_train_stats = {}

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}", v, epoch)
                    for i, v in enumerate(test_train_stats[k]):
                        self.writer.add_scalar(f"Train/{k}_{i}", v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat["epoch"] == epoch and self.output_dir:
                    dist_utils.save_on_master(
                        self.state_dict(), self.output_dir / "best.pth"
                    )

            print(f"best_stat: {best_stat}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"train_{k}": v for k, v in test_train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
