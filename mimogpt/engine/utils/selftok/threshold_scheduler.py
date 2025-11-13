from mimogpt.models.selftok.models_ours import Encoder

class ThresholdScheduler:
    def __init__(
        self,
        init_threshold,
        final_threshold,
        constant_step=4000,
        end_step=20000,
    ):
        assert end_step > constant_step
        self.init_threshold = init_threshold
        self.final_threshold = final_threshold
        self.constant_step = constant_step
        self.end_step = end_step
        self.fn = max if final_threshold <= init_threshold else min
        
    def step(self, encoder, cur_iter):
        if cur_iter <= self.constant_step:
            return
        rate = (self.init_threshold-self.final_threshold) / (self.end_step-self.constant_step)
        cur_threshold = self.init_threshold - rate * (cur_iter-self.constant_step)
        cur_threshold = self.fn(self.final_threshold, cur_threshold)
        encoder.quantizer._codebook.threshold_ema_dead_code = cur_threshold
        


class ObjectiveScheduler:
    def __init__(self, start_iter, end_iter):
        self.start = start_iter
        self.end = end_iter

    def step(self, tokenizer, cur_iter):
        if cur_iter <= self.start:
            return
        if cur_iter >= self.end:
            return
        delta = cur_iter - self.start
        rate = 1. / (self.end - self.start)
        tokenizer.recon_ratio = 1. - rate * delta
        return
    

class LowResDroprateScheduler:
    def __init__(self, start_iter=0, end_iter=0, start_rate=0.0, end_rate=0.0):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_rate = start_rate
        self.end_rate = end_rate
        if self.end_iter != self.start_iter:
            self.r = (self.start_rate - self.end_rate) / (self.end_iter - self.start_iter)
        else:
            self.r = 0

    def step(self, mmdit, cur_iter):
        if cur_iter < self.start_iter:
            mmdit.low_res_drop_rate = 1.0
        elif cur_iter < self.end_iter:
            mmdit.low_res_drop_rate = self.start_rate - self.r * (cur_iter - self.start_iter)
        else:
            mmdit.low_res_drop_rate = self.end_rate
        if cur_iter % 50 == 0:
            print("cur_iter", cur_iter, "mmdit.low_res_drop_rate", mmdit.low_res_drop_rate)
        return