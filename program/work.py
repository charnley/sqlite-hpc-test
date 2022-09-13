import hashlib
import logging
import os
import tempfile
from pathlib import Path
from time import time

import pandas as pd

from program import models
from program.utils.reading import read_inputfile

logger = logging.getLogger(__name__)
filenames = tempfile._get_candidate_names()


def do_some_work(row: dict, work_column="name") -> dict:
    m = hashlib.md5()
    value: str = row[work_column]
    m.update(value.encode())
    work = hashlib.md5(value.encode()).hexdigest()
    row["work"] = work
    return row


def main(args=None):

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--filename", action="store", help="", metavar="FILE", type=Path
    )
    parser.add_argument(
        "--sqlite",
        action="store",
        help="",
        metavar="FILE",
        type=Path,
        default=Path("./database.sqlite"),
    )
    parser.add_argument(
        "--postgres",
        action="store",
        help="Hostname of postgres instance",
        metavar="HOSTNAME",
    )
    args = parser.parse_args(args)

    # Read work
    filename: Path = args.filename
    assert filename.is_file(), f"{filename} does not exist"
    pdf = read_inputfile(filename)

    # Connect to database
    database: Path = args.sqlite
    assert database.is_file(), f"Could not read sqlite file: {database}"

    if args.postgres:

        kwargs = {
            "username": os.environ.get("USER"),
            "hostname": args.postgres,
            "database_name": "postgres",
        }

        logger.info(f"Trying to connect to {args.postgres}")
        session = models.get_session_pg(**kwargs)
    else:
        session = models.get_session(args.sqlite)

    # Init before work
    # and use update?

    for row in pdf.to_dict(orient="records"):

        row = do_some_work(row)

        kwargs = dict(hashkey=row["work"], name=row["name"])
        item = models.Prediction(**kwargs)

        time_a = time()
        session.add(item)
        session.commit()
        time_b = time()
        logger.info(f"Store took {time_b-time_a:.3f} sec")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
