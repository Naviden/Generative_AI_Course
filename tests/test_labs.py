import json

import numpy as np
import pytest
import torch

from genai_course.evaluation import paired_bootstrap_interval, sliced_wasserstein
from genai_course.labs import main, mixture


@pytest.mark.parametrize("lab", ["attention", "vae", "flow", "dpo"])
def test_lab_runs_and_records_reproducible_results(lab, tmp_path):
    args = [lab, "--steps", "8", "--output", str(tmp_path), "--seed", "7"]
    first, second = main(args), main(args)
    assert first != second
    a, b = [json.loads((p / "metrics.json").read_text()) for p in [first, second]]
    assert a["metrics"] == b["metrics"]
    assert a["environment"]["device"] == "cpu"
    assert "objectives.py" in a["source_sha256"]
    assert a["config"]["seed"] == 7
    if lab in ("vae", "flow"):
        assert a["data_sha256"]["train"] != a["data_sha256"]["heldout"]
        arrays = np.load(first / "arrays.npz")
        assert arrays["generated"].shape == (256, 2)
        assert np.isfinite(arrays["generated"]).all()
    if lab == "dpo":
        assert a["metrics"]["final_probabilities"][0] > 1 / 3
        assert a["metrics"]["loss_last"] < a["metrics"]["loss_first"]


def test_sampling_budget_holds_training_fixed(tmp_path):
    common = ["flow", "--steps", "10", "--seed", "3", "--output", str(tmp_path)]
    first = main(common + ["--sampling-steps", "5"])
    second = main(common + ["--sampling-steps", "50"])
    a, b = [np.load(p / "arrays.npz") for p in [first, second]]
    np.testing.assert_array_equal(a["train_objective"], b["train_objective"])
    np.testing.assert_array_equal(a["heldout"], b["heldout"])
    assert not np.array_equal(a["generated"], b["generated"])


@pytest.mark.parametrize("args", [["--steps", "0"], ["--beta", "-1"], ["--beta", "nan"], ["--seed", "-1"], ["--sampling-steps", "0"]])
def test_invalid_cli_rejected(args, tmp_path):
    with pytest.raises(SystemExit):
        main(["flow", "--output", str(tmp_path)] + args)


def test_synthetic_data_reproducibility():
    torch.testing.assert_close(mixture(20, 1), mixture(20, 1))
    assert not torch.equal(mixture(20, 1), mixture(20, 2))


def test_sliced_w1_properties():
    a = np.random.default_rng(1).normal(size=(20, 2))
    assert sliced_wasserstein(a, a) == 0
    assert sliced_wasserstein(a, a[::-1]) == 0
    assert sliced_wasserstein(a, a + 5) > 0
    assert sliced_wasserstein(a, a + 5) == sliced_wasserstein(a + 5, a)
    with pytest.raises(ValueError):
        sliced_wasserstein(a, a[:-1])


def test_bootstrap_known_paired_difference():
    assert paired_bootstrap_interval([2, 3, 4], [1, 2, 3]) == (1., 1.)
    assert paired_bootstrap_interval([1, 2, 3], [2, 3, 4]) == (-1., -1.)
    with pytest.raises(ValueError):
        paired_bootstrap_interval([1], [1])
