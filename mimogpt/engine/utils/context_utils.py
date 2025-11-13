import os
from contextlib import ContextDecorator
from typing import Any

try:
    import moxing as mox

    mox.file.set_auth(is_secure=False)
except ImportError:
    mox = None


class MemartsCopyContext(ContextDecorator):
    def __init__(self):
        """
        This context manager is only in use when memarts is enabled, if memarts is not enabled it does nothing.
        It basically set _USE_MEMARTS to False when enter this context, then set _USE_MEMARTS back to True when exit
        Because normal mox copy in main process will cause error in dataloader when memarts is enabled,
        so we need to use this context manager to wrap any mox copy call to avoid error.

        To use this context manager:
        with MemartsCopyContext():
            mox.file.copy(xxx, xx)

        or

        @MemartsCopyContext()
        def mox_copy(src, dst):
            mox.file.copy_parallel(src, dst)
        """
        self.use_memarts = (os.environ.get("USE_MEMARTS") == "1") and (mox is not None)

    def __enter__(self):
        if self.use_memarts:
            mox.file.file_io._USE_MEMARTS = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if self.use_memarts:
            mox.file.file_io._USE_MEMARTS = True
