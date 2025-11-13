# -*- coding: utf-8 -*-


from .train_loop import HookBase
import torch
import copy
import os
from .cloud_copy import mox_copy
from mimogpt.utils import hf_logger


def devices():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class EMAhook(HookBase):
    def __init__(self, cfg, is_root):
        self.cfg = cfg
        self.use_ema = self.cfg.optimize.use_ema
        self.ema_test_interval = self.cfg.optimize.ema_test_interval
        self.ema_factor = self.cfg.optimize.ema_factor
        self.ema_in_cpu = self.cfg.optimize.ema_in_cpu
        self.gradient_accumulation_steps = self.cfg.optimize.get("gradient_accumulation_steps", 1)

        # self.Trainer = Trainer

        self.is_root = is_root
        self.save_interval = int(self.cfg.common.save_per_epochs * self.cfg.dataloader_len)
        self.delete_after_upload = cfg.common.get("delete_after_upload", False)
        self.output_path = self.cfg.common.output_path

    def after_step(self):
        # print((self.trainer.iter + 1))
        # print(self.use_ema)
        if self.use_ema and (self.trainer.iter + 1) % self.gradient_accumulation_steps == 0:
            with torch.no_grad():
                for p, p_ema in zip(self.trainer.model.parameters(), self.trainer.ema_model.parameters()):
                    if self.ema_in_cpu:
                        p1 = p.data.detach().clone().cpu()
                    else:
                        p1 = p.data.detach().clone()
                    p_ema.data.mul_(self.ema_factor).add_((1 - self.ema_factor) * p1)

        if self.trainer.iter % self.save_interval == (self.save_interval - 1):
            if self.cfg.optimize.use_ema:
                save_name = "ema_iter_%d.pth" % (self.trainer.iter)
                self.save_model(self.trainer.ema_model.state_dict(), save_name)
                torch.cuda.empty_cache()

    def before_train(self):
        if self.use_ema:
            print("Build EMA Model...")
            if self.ema_in_cpu:
                self.trainer.ema_model = copy.deepcopy(self.trainer.model_before_ddp)
            else:
                self.trainer.ema_model = copy.deepcopy(self.trainer.model_before_ddp).to(devices())
            self.trainer.ema_model.requires_grad_(False)
            self.trainer.ema_model.eval()
            print("Done...")

    @property
    def save_path(self):
        if self.__dict__.get("_output_path", None) is None:
            output_path = os.path.join(self.output_path, "ckpt")
            os.makedirs(output_path, exist_ok=True)
            self.__dict__["_output_path"] = output_path
        return self.__dict__["_output_path"]

    def save_model(self, checkpoint, save_name):
        if self.use_ema:
            if self.is_root:
                local_weights = os.path.join(self.save_path, save_name)
                torch.save(checkpoint, local_weights)
                try:
                    import moxing as mox

                    roma_weights_fp = os.path.join(self.cfg.train_url, local_weights)
                    roma_weights_dirname = os.path.dirname(roma_weights_fp)
                    if not mox.file.exists(roma_weights_dirname):
                        mox.file.make_dirs(roma_weights_dirname)
                    mox_copy(local_weights, roma_weights_fp)
                    hf_logger.info("save weight success, roma_weights_fp:{}".format(roma_weights_fp))
                    if self.delete_after_upload:
                        os.remove(local_weights)
                        hf_logger.info(f"{local_weights} removed")
                except:
                    hf_logger.info("save weight success, local_weights_fp:{}".format(local_weights))
