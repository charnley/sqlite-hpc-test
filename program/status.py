import logging
import os
from typing import List

import pandas as pd

from program import models

logger = logging.getLogger(__name__)


def rows2pdf(rows: List[models.Prediction]) -> pd.DataFrame:

    rows_ = []

    for row in rows:
        row_ = dict((col, getattr(row, col)) for col in row.__table__.columns.keys())
        rows_.append(row_)

    return pd.DataFrame(rows_)


def main(args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postgres",
        action="store",
        help="Hostname of postgres instance",
        metavar="HOSTNAME",
    )
    args = parser.parse_args(args)

    if args.postgres:

        kwargs = {
            "username": os.environ.get("USER"),
            "hostname": args.postgres,
            "database_name": "postgres",
        }

        logger.info(f"Trying to connect to {args.postgres}")
        session = models.get_session_pg(**kwargs)

        query = session.query(models.Prediction)

        print(query.count())

        entry = query.first()
        print(entry.__dict__)

        entries = query.limit(5).all()
        pdf = rows2pdf(entries)

        print(pdf)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
