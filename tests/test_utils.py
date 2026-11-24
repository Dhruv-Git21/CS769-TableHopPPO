import json
import random

import numpy as np
import torch

from tablehop.utils.config import get_config
from tablehop.utils.device import get_device
from tablehop.utils.logging import CSVLogger
from tablehop.utils.runmeta import new_run_dir
from tablehop.utils.seeding import set_seed


def test_config_inheritance_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: 3\n")
    (tmp_path / "c.yaml").write_text("base: base.yaml\na: {y: 5}\n")
    cfg = get_config(tmp_path / "c.yaml", ["a.x=10", "b=7", "new.k=hi"])
    assert cfg["a"] == {"x": 10, "y": 5} and cfg.b == 7 and cfg.new.k == "hi"


def test_repo_configs():
    cfg = get_config("configs/smoke_synthetic.yaml")
    assert cfg["reward"]["w_hop"] == 0.2 and cfg["ppo"]["clip_eps"] == 0.2


def test_seeding():
    def draw():
        set_seed(3)
        return random.random(), np.random.rand(), torch.rand(1).item()
    assert draw() == draw()


def test_device():
    assert get_device("auto") in ("cuda", "mps", "cpu")
    assert get_device("cpu") == "cpu"


def test_run_dir_meta(tmp_path):
    rm = new_run_dir("ppo", "synth", 1, {"device": "cpu", "k": 1}, tmp_path)
    m = json.loads((rm.run_dir / "meta.json").read_text())
    for k in ("seed", "git_hash", "config", "device", "status", "git_dirty", "start_time"):
        assert k in m
    assert m["status"] == "running" and m["seed"] == 1
    rm.finish("done")
    assert json.loads((rm.run_dir / "meta.json").read_text())["status"] == "done"
    lg = CSVLogger(rm.run_dir)
    lg.log({"step": 0, "loss": 1.0})
    assert (rm.run_dir / "train_log.csv").read_text().startswith("step,loss")
