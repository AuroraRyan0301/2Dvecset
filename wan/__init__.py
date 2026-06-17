# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

from . import configs, distributed, modules

# Task pipelines are imported optionally: each pulls task-specific deps (s2v->whisper,
# animate->sam2/peft, ...). A missing optional dep for one task must not block importing the
# package or other tasks (e.g. the OBB task). Pipelines still load normally when their deps exist.
for _name, _cls in [("image2video", "WanI2V"), ("speech2video", "WanS2V"),
                    ("text2video", "WanT2V"), ("textimage2video", "WanTI2V"),
                    ("animate", "WanAnimate")]:
    try:
        globals()[_cls] = getattr(__import__(f"{__name__}.{_name}", fromlist=[_cls]), _cls)
    except Exception as _e:  # noqa: BLE001
        warnings.warn(f"[wan] task pipeline '{_name}' unavailable (optional deps missing): {_e}")