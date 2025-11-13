import os
import torch
import torch.distributed as dist
from torch.profiler import profile, schedule
from .timeline_analysis import parse_timeline


class Nothing(object):
    def __init__(self, *args, **kwargs):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def start(self):
        return

    def stop(self):
        return

    def step(self):
        return

    def export_chrome_trace(self, path):
        return


def trace_handler(prof, train_url, name="train"):
    if not os.path.exists("pytorch_profile"):
        os.makedirs("pytorch_profile")

    src = os.path.join("pytorch_profile", "timeline_{}_step_{}.json".format(name, prof.step_num))
    prof.export_chrome_trace(src)
    analysis_ret = parse_timeline(src)

    try:
        import moxing as mox

        if train_url is not None:
            dst = os.path.join(train_url, src)
            dst_dirname = os.path.dirname(dst)
            if not mox.file.exists(dst_dirname):
                mox.file.make_dirs(dst_dirname)
            mox.file.copy(src, dst)
            print("save profile timeline from {} to {}".format(src, dst))

            for name in analysis_ret:
                dst = os.path.join(train_url, name)
                mox.file.copy(name, dst)
                print("save profile files from {} to {}".format(name, dst))
        else:
            print("skip backup profile timeline since train_url=None")
    except:
        pass


def get_profile_fn(args):
    if dist.get_rank() == 0 and args.profile:
        my_schedule = schedule(
            skip_first=args.profile_skip_first,
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=args.profile_repeat,
        )
        profile_fn = profile
    else:
        my_schedule = None
        profile_fn = Nothing()
    return my_schedule, profile_fn


def print_memory_status(msg, empty_cache=True, is_print=True):
    if empty_cache:
        torch.cuda.empty_cache()

    mem_res = torch.cuda.memory_reserved() / (1024**3)
    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
    if is_print:
        print("{}, memory_reserved={:.2f}G, memory_allocated={:.2f}G".format(msg, mem_res, mem_alloc))

    return mem_res, mem_alloc
