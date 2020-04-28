import logging
from argparse import Namespace
from typing import Iterator
from pathlib import Path
import pickle as pkl
from datetime import datetime
from typing import Dict, Any, Tuple, Set

import torch
from joblib import Parallel, delayed
from hyperopt import hp, tpe, fmin, Trials

from egg.zoo.visA import game

logger = logging.getLogger()
ConfigDiff = Dict[str, Tuple[Any, Any]]

### General config ###

N_JOBS = 1
LOG_FILE = "log.txt"
# Config ids to skip
should_skip: Set[int] = set({})
# bayes_opt or grid_search
TASK = "bayes_opt"
logger.setLevel(logging.WARNING)

DEFAULT_OPTS = Namespace(
    batch_size=32,
    checkpoint_dir=None,
    checkpoint_freq=0,
    cuda=False,
    data_path="data/visa/US",
    data_set="visa",
    device=torch.device(type="cpu"),
    dump_data=None,
    dump_output=None,
    early_stopping_thr=0.98,
    examples_per_epoch=1000,
    force_eos=False,
    load_from_checkpoint=None,
    lr=0.001,
    max_len=5,
    n_classes=None,
    n_distractors=9,
    n_epochs=4,
    no_cuda=True,
    optimizer="adam",
    preemptable=False,
    random_seed=0,
    receiver_cell="gru",
    receiver_embedding=50,
    receiver_entropy_coeff=0.01,
    receiver_hidden=20,
    receiver_layers=1,
    receiver_lr=0.001,
    sender_cell="gru",
    sender_embedding=50,
    sender_entropy_coeff=0.01,
    sender_hidden=20,
    sender_layers=1,
    sender_lr=0.001,
    temperature=1.0,
    tensorboard=False,
    tensorboard_dir="runs/",
    toposim_embed=True,
    train_mode="gs",
    valid_prop=0.2,
    validation_freq=1,
    vocab_size=10,
)

MAX_EVALS = 100
SEARCH_SPACE = {
    "train_mode": hp.choice("train_mode", ["gs", "rf"]),
    "max_lens": hp.uniformint("max_lens", 1, 10),
    "vocab": hp.uniformint("vocab", 10, 200),
    "lr": hp.uniform("lr", 0.00001, 0.0001),
    "sender_hidden": hp.uniformint("sender_hidden", 10, 100),
    "receiver_hidden": hp.uniformint("receiver_hidden", 10, 100),
}


### End Config ###

# (idx, log dir, opts, opts diff)
RunArgs = Tuple[int, Path, Namespace, ConfigDiff]


def copy(ns: Namespace) -> Namespace:
    """Peroform a shallow copy on the given namespace."""
    return Namespace(**vars(ns))


def ns_diff(x: Namespace, y: Namespace) -> ConfigDiff:
    diff = {}
    for k, v in x._get_kwargs():
        if getattr(y, k) != getattr(x, k):
            diff[k] = (getattr(x, k), getattr(y, k))
    return diff


def opt_generator(log_dir: Path, base_opts: Namespace) -> Iterator[RunArgs]:
    counter = 0
    for max_len in [1, 2, 4, 8]:
        for vocab_size in [2, 4, 8, 16, 32]:
            for n_distractors in [4, 9, 19]:
                for train_mode in ["rf", "gs"]:
                    opts = copy(base_opts)
                    opts.max_len = max_len
                    opts.n_distractors = n_distractors
                    opts.vocab_size = vocab_size
                    opts.train_mode = train_mode
                    diff = ns_diff(base_opts, opts)
                    # If there are specific configs that shouldn't be run, they will be
                    # skipped
                    if counter not in should_skip:
                        yield (counter, log_dir, opts, diff)
                    counter += 1


def run_config(args: RunArgs) -> Dict[str, Any]:
    torch.set_num_threads(1)

    idx, log_dir, opts, diff = args
    output = game.run_game(opts)

    max_acc_v_idx = output["valid"]["acc"].argmax()
    sum_list = [
        f"diff: {diff}",
        f"objective: {output['objective']}",
        f"max v acc: {output['valid']['acc'][max_acc_v_idx]}",
        f"ts @ max v acc: {output['valid']['toposim'][max_acc_v_idx]}",
        f"epoch @ max v acc: {max_acc_v_idx}",
    ]
    summary = "\n".join(f"{idx}: {item}" for item in sum_list)
    print(summary)
    print()
    with (log_dir / LOG_FILE).open("a") as log_file:
        log_file.write(summary + "\n")
    with (log_dir / f"config_{idx}.pkl").open("wb") as pkl_file:
        pkl.dump(output, pkl_file)
    return output


def main() -> None:
    # TODO When we run the game here, we are skipping the "important" intialization
    # of the EGG framework, but this involves editing global state which is horrible
    # for doing things programmatically. If something doesn't get initialized, this is
    # probably why.

    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir()
    timestamp = datetime.strftime(datetime.today(), "%Y-%m-%d_%H-%M-%S")
    log_dir /= timestamp
    log_dir.mkdir()
    with (log_dir / LOG_FILE).open("a") as logfile:
        logfile.write(str(DEFAULT_OPTS) + "\n")
    with (log_dir / f"config_default.pkl").open("wb") as pkl_file:
        pkl.dump(DEFAULT_OPTS, pkl_file)

    if TASK == "grid_search":
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(run_config)(opts) for opts in opt_generator(log_dir, DEFAULT_OPTS)
        )
    elif TASK == "bayes_opt":
        # RunArgs = Tuple[int, Path, Namespace, ConfigDiff]
        counter = 0

        def run_wrapper(changed_opts: Dict[str, Any]) -> float:
            new_opts = copy(DEFAULT_OPTS)
            for k, v in changed_opts.items():
                setattr(new_opts, k, v)
            diff = ns_diff(DEFAULT_OPTS, new_opts)
            nonlocal counter
            output = run_config((counter, log_dir, new_opts, diff))
            counter += 1
            return output["objective"]

        print(f"Initial parameters: {DEFAULT_OPTS}")
        print(f"Bayesian optimization over: {SEARCH_SPACE}")

        hypopt_trials = Trials()
        best_params = fmin(
            run_wrapper,
            SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=hypopt_trials,
        )
        print(best_params)
    else:
        raise ValueError(f"Unknown task: {TASK}")


if __name__ == "__main__":
    main()
