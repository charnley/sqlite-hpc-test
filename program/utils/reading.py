import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_inputfile(filename: Path) -> Optional[pd.DataFrame]:

    if filename.suffix == ".pkl":
        pdf: pd.DataFrame = pd.read_pickle(filename)

    elif filename.suffix == ".csv":
        options = dict(
            sep=",",
            engine="python",
            skipinitialspace=True,  # skip spaces after delimter
            skip_blank_lines=True,
        )
        pdf: pd.DataFrame = pd.read_csv(filename, **options)

    else:

        logger.error(f"Unable to parse {filename}")
        return None

    return pdf


def write_csv(filename: Path, df: pd.DataFrame):

    kwargs = dict(header=True, index=False,)

    df.to_csv(filename, **kwargs)
