"""
Generate a task array on cluster to push everything into sqlite
"""
import itertools
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from program import models
from program.utils import uge
from program.utils.reading import write_csv

logger = logging.getLogger(__name__)

CMD_WORKER = "python -m program.work"


def split_jobs(iterable, n_entries: int, n_jobs: int):
    """ Yield dataframe of content"""

    work_per_job = int(np.ceil(n_entries / n_jobs))  # type: ignore

    for _ in range(n_jobs):
        rows = list(itertools.islice(iterable, work_per_job))
        yield pd.DataFrame(rows)


def generate_work(n_entries):

    names = tempfile._get_candidate_names()  # type: ignore
    for _ in range(n_entries):
        name = next(names)

        row = {
            "name": name,
            "cool": "something",
        }
        yield row


def main(args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", action="store", help="", metavar="FILE")
    parser.add_argument(
        "--sqlite",
        action="store",
        help="",
        metavar="FILE",
        type=Path,
        default=Path("./database.sqlite"),
    )
    parser.add_argument(
        "--n-entries", action="store", help="", metavar="INT", type=int, default=1
    )
    parser.add_argument(
        "--n-jobs", action="store", help="", metavar="INT", type=int, default=1
    )
    parser.add_argument("--scr", action="store", default="./_scr", type=Path)
    args = parser.parse_args(args)

    # Read work
    args.scr.mkdir(exist_ok=True, parents=True)
    assert args.scr.is_dir(), f"{args.scr} does not exist"

    # Create session, init file if not exist
    _ = models.get_session(args.sqlite, init=True)

    # Generate work
    filename = "__work__.{n}.csv"
    for n, work in enumerate(
        split_jobs(generate_work(args.n_entries), args.n_entries, args.n_jobs)
    ):
        _filename = filename.format(n=n + 1)
        logger.info(f"Writing '{args.scr / _filename}'")
        write_csv(args.scr / _filename, work)

    # Generate submission script
    worker_options = dict(
        filename=filename.format(n=uge.UGE_TASK_ID), sqlite=args.sqlite.resolve(),
    )

    # Hack
    environ = dict(
        PYTHONPATH=Path(".").resolve()
    )

    worker_args = uge.get_argument_string(worker_options)
    script = uge.generate_script(
        cmd=f"{CMD_WORKER} {worker_args}", cwd=args.scr.resolve(), environ=environ,task_stop=args.n_jobs,
    )

    # Submitting and waiting for uge work
    uge.distribute_work([script], scr=args.scr, wait=True, wait_time=5)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
