import logging
from pathlib import Path
from time import time

from program import models, work

logger = logging.getLogger(__name__)

__version__ = 1.0


def main(args=None):

    logging.basicConfig(level=logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("-n", "--number-of-jobs", action="store", default=1, type=int)
    args = parser.parse_args(args)

    logging.info("starting")

    session = models.get_session(Path("./database.sqlite"))
    records = session.query(models.Prediction).all()

    logger.info(f"Found {len(records)} entries in database")

    for _ in range(args.number_of_jobs):

        results = work.generate_prediction()

        kwargs = dict(name=results["name"], status="Started")

        item = models.Prediction(**kwargs)

        time_a = time()
        session.add(item)
        session.commit()
        time_b = time()
        logger.info(f"Store took {time_b-time_a:.3f}s")


if __name__ == "__main__":
    main()
