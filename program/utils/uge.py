import argparse
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from jinja2 import Template
from pandas.core.frame import DataFrame

from .linesio import get_index

_logger = logging.getLogger(__name__)

FILENAMES = tempfile._get_candidate_names()

SCRIPT_TEMPLATE_JINJA2 = """#!/bin/bash

#$ -N {{ name }}
#$ -cwd
#$ -V
#$ -l h_rt={{ hours }}:{{ mins | default("00", true) }}:00
#$ -l m_mem_free={{ mem }}G
#$ -pe smp {{ cores }}{% if task_stop %}
#$ -t {{ task_start | default(1, true) }}-{{ task_stop }}:{{ task_step | default(1, true) }}
#$ -tc {{ task_concurrent }}{% endif %}
#$ -o {{ log_dir | default("./ugelogs", true) }}/
#$ -e {{ log_dir | default("./ugelogs", true) }}/

{% if module_purge %}module purge\n{% endif %}\
{% for dir in module_use %}module use {{ dir }}\n{% endfor %}\
module load{% for module in module_load %} {{ module }}{% endfor %}
# Set cores
CORES={{ cores }}
export OPENBLAS_NUM_THREADS=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES

{% if cwd %}cd {{ cwd }}\n\n{% endif %}\
{% for key, value in environ.items() %}export {{ key }}={{ value }}\n{% endfor %}\

{{ cmd }}
"""


TEMPLATE_TASKARRAY = """#!/bin/bash

#$ -N {{ name }}
#$ -cwd
#$ -V
#$ -l h_rt={{ hours }}:{{ mins | default("00", true) }}:00
#$ -l m_mem_free={{ mem }}G
#$ -pe smp {{ cores }}{% if task_stop %}
#$ -t {{ task_start | default(1, true) }}-{{ task_stop }}:{{ task_step | default(1, true) }}
#$ -tc {{ task_concurrent }}{% endif %}
#$ -o {{ log_dir | default("./ugelogs", true) }}/
#$ -e {{ log_dir | default("./ugelogs", true) }}/

# Set cores
CORES={{ cores }}
export OPENBLAS_NUM_THREADS=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES

{% if cwd %}cd {{ cwd }}\n\n{% endif %}\
{% for key, value in environ.items() %}export {{ key }}={{ value }}\n{% endfor %}\

{{ cmd }}
"""


UGE_TASK_ID = "$SGE_TASK_ID"

UGE_COMMAND_SUBMIT = "qsub {filename}"
UGE_COMMAND_SUBMIT_SYNC = "qsub -sync y {filename}"
UGE_COMMAND_SHELL = (
    "qrsh -N '{name}' -cwd -V -verbose -l "
    "mem_free={mem}G,h_rt={hours}:00:00 -pe smp {cores} 'bash {filename}'"
)

UGE_TASK_CONCURRENT = 250

HERE = "./"

DEFAULT_LOGS_PATH = Path("ugelogs")


def generate_taskarray_script(
    cmd: str,
    environ: Dict[str, str] = {},
    log_dir: Path = DEFAULT_LOGS_PATH,
    n_cores: int = 4,
    n_hours: int = 12,
    n_mem: int = 4,
    name: str = "UGEJob",
    task_concurrent: int = 100,
    task_start: int = 1,
    task_step: int = 1,
    task_stop: Optional[int] = None,
):
    """

    Args:
        cmd (str): 
        environ (Dict[str, str]): 
        log_dir (Path): 
        n_cores (int): 
        n_hours (int): 
        n_mem (int): 
        name (str): 
        task_concurrent (int): 
        task_start (int): 
        task_step (int): 
        task_stop (Optional[int]): 

    Returns:
        script (str)
    """

    assert n_cores > 0
    assert n_hours > 0
    assert log_dir.is_dir()

    kwargs = dict(
        cmd=cmd,
        environ=environ,
        log_dir=log_dir,
        n_cores=n_cores,
        n_hours=n_hours,
        n_mem=n_mem,
        name=name,
        task_concurrent=task_concurrent,
        task_start=task_start,
        task_step=task_step,
        task_stop=task_stop,
    )

    template = Template(TEMPLATE_TASKARRAY)
    msg = template.render(**kwargs)

    return msg


# pylint: disable=too-many-arguments,too-many-locals,dangerous-default-value
def generate_script(
    cmd: str,
    cores: int = 4,
    cwd: Optional[Path] = None,
    environ: Dict[str, str] = {},
    hours: int = 12,
    log_dir: Optional[Path] = None,
    mem: int = 4,
    mins: int = 0,
    module_load: List[str] = [],
    module_purge: bool = False,
    module_use: List[str] = [],
    name: str = "UGEJob",
    task_concurrent: int = 100,
    task_start: int = 1,
    task_step: int = 1,
    task_stop: Optional[int] = None,
) -> str:
    # pylint: disable=unused-argument

    assert isinstance(cores, int), f"Setting cores to {cores}, is not a number"

    kwargs = locals()
    template = Template(SCRIPT_TEMPLATE_JINJA2)
    msg = template.render(**kwargs)

    return msg


# pylint: disable=dangerous-default-value
def submit(
    submit_script: str,
    scr: Union[str, Path] = HERE,
    filename: Optional[str] = None,
    cmd: str = UGE_COMMAND_SUBMIT,
    cmd_options: Dict[str, str] = {},
    dry: bool = False,
) -> Optional[str]:
    """ submit script and return UGE ID """

    scr = Path(scr)
    scr.mkdir(parents=True, exist_ok=True)

    if filename is None:
        random_key = generate_name()
        filename = f"tmp_uge.{random_key}.sh"

    with open(scr / filename, "w") as f:
        f.write(submit_script)

    _logger.debug(f"Writing {filename} for UGE on {scr}")

    # TODO Should be random name to avoid raise-condition
    cmd = cmd.format(filename=filename, **cmd_options)
    _logger.debug(cmd)
    _logger.debug(scr)

    # Dry run means Dont execute
    if dry:
        return None

    stdout, stderr = execute(cmd, cwd=scr)

    if stderr:
        for line in stderr.split("\n"):
            _logger.error(line)
        return None

    # Successful submission
    # find id
    lines = stdout.strip().rstrip().split("\n")
    for line in lines:
        _logger.debug(line)

    line_idx = get_index(lines, "has been submitted")

    if line_idx is None:
        _logger.error("Could not find job id from UGE submission")
        return None

    # NOTE sometimes there is random stuff from qsub stdout
    # Line format: Your job JOB_ID ("JOB_NAME") has been submitted
    line = lines[line_idx]
    uge_id = line.split()[2].split(".")[0]

    _logger.info(f"Job {uge_id} has been submitted to UGE")

    return uge_id


def delete(job_id: Union[str, int]) -> None:

    # TODO denied: job "6905376" does not exist

    cmd = f"qdel {job_id}"
    _logger.debug(cmd)

    stdout, stderr = execute(cmd)

    for line in stderr.split("\n"):
        _logger.error(line)

    for line in stdout.split("\n"):
        _logger.info(line)


def get_status(job_id: Union[str, int]) -> Optional[Dict[str, str]]:
    """
    Get status of SGE Job

    job_state
        q - queued
        qw - queued wait ?
        r - running
        dr - deleting
        dx - deleted

    job_name

    """

    cmd = f"qstat -j {job_id}"
    # TODO Check syntax for task-array

    stdout, stderr = execute(cmd)

    stderr = stderr.replace("\n", "")
    stderr = stderr.strip().rstrip()

    if stderr:
        _logger.debug(stderr)
        return None

    lines = stdout.split("\n")

    status_ = dict()

    for line in lines:

        line_ = line.split(":")
        if len(line_) < 2:
            continue

        # TODO Is probably task_array related and needs are more general fix
        # job_state             1:    r
        key = line_[0]
        key = key.replace("    1", "")
        key = key.strip()

        content = line_[1].strip()

        status_[key] = content

    return status_


def execute(
    cmd: str,
    cwd: Optional[Path] = None,
    shell: bool = True,
    timeout: Optional[int] = None,
) -> Tuple[str, str]:
    """Execute command in directory, and return stdout and stderr

    :param cmd: The shell command
    :param cwd: Change directory to work directory
    :param shell: Use shell or not in subprocess
    :param timeout: Stop the process at timeout (seconds)
    :returns: stdout and stderr as string
    """

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    ) as process:

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _logger.error("Process timed out")
            stdout = ""
            stderr = ""

    return stdout, stderr


def wait_for_jobs(jobs: List[str], respiratory: int = 60):
    """ Monitor UGE jobs and yield when jobs are done """

    _logger.info(f"Submitted {len(jobs)} job(s) to UGE cluster")

    start_time = time.time()

    while len(jobs):

        _logger.info(
            f"... and breathe for {respiratory} sec, still running {len(jobs)} job(s)"
        )
        time.sleep(respiratory)

        for job_id in jobs:

            job_is_done = _uge_is_job_done(job_id)

            if job_is_done:
                yield job_id
                jobs.remove(job_id)

    end_time = time.time()
    diff_time = end_time - start_time

    _logger.info(f"All jobs finished and took {diff_time/60/60:.2f}h")


def _uge_is_job_done(job_id: str) -> bool:

    # TODO works for task-array?

    still_waiting_states = ["q", "r", "qw", "dr"]

    status = get_status(job_id)

    if status is None:
        return True

    state = status.get("job_state", "qw")
    _logger.debug(f"uge {job_id} is {state}")

    if state not in still_waiting_states:
        return True

    return False


def uge_log_error(filename: Path, name: str = "uge") -> None:

    if not filename.exists():
        _logger.error(f"could not read {filename}")
        return

    with open(filename, "r") as f:
        for line in f:
            line = line.strip().rstrip()
            # ignore empty lines
            if not line:
                continue
            _logger.error(f"{name} - {line}")


def uge_log_info(filename: Path, name: str = "uge") -> None:

    if not filename.exists():
        _logger.error(f"could not read {filename}")
        return

    with open(filename, "r") as f:
        for line in f:
            line = line.strip().rstrip()
            # ignore empty lines
            if not line:
                continue
            _logger.info(f"{name} - {line}")


def generate_name():
    """ Generate a random name"""
    return next(FILENAMES)


def reconstruct_environment():
    """ Not implemented"""
    raise NotImplementedError
    # TODO Pythonpath
    # TODO lmod modules paths and purge
    # TODO conda, venv etc


def distribute_work(
    scripts: List[str],
    scr: Path,
    wait: bool = False,
    wait_time = 60,
    cmd=UGE_COMMAND_SUBMIT,
    logdir=DEFAULT_LOGS_PATH,
):

    # TODO Log path could be somewhere else than scracth

    job_ids = list()

    for script in scripts:

        # Submit script
        job_id = submit(script, scr=scr, cmd=cmd)
        job_ids.append(job_id)

    if wait:
        finished_ids = wait_for_jobs(job_ids, respiratory=wait_time)

        for job_id in finished_ids:
            read_logfiles(scr / logdir, job_id)

        _logger.info("Finished UGE jobs")


def read_logfiles(log_path, job_id):
    """ Check for errors in log files """
    # Read std error and out
    stderr_filenames = log_path.glob(f"*.e{job_id}*")
    stdout_filenames = log_path.glob(f"*.o{job_id}*")
    for filename in stderr_filenames:
        uge_log_error(filename, name=str(job_id))
    for filename in stdout_filenames:
        uge_log_info(filename, name=str(job_id))


def get_options(args: argparse.Namespace):

    # "cwd": None,
    # "environ": variables,
    # "log_dir": "log",
    # "module_load": env_info.lmod_modules,
    # "module_use": env_info.lmod_paths,
    # "module_purge": True,
    # "task_stop": total_tasks,
    # "task_concurrent": uge_utils.UGE_TASK_CONCURRENT,

    # options = vars(args)
    # uge_options = {k:v for k,v in options.items() if "uge_" in k}

    # Parse time
    hours, mins = parse_time(args.uge_time)
    assert isinstance(hours, int), "Unable to parse time limit for UGE job"
    assert isinstance(mins, int), "Unable to parse time limit for UGE job"

    uge_options = {
        "hours": hours,
        "mins": mins,
        "name": args.uge_name,
        "cores": args.uge_n_cores,
    }

    return uge_options


def add_arguments(parser, n_cores=4, time="10hours", name="molpipe", per_job=100):

    group = parser.add_argument_group("uge")

    group.add_argument(
        "--uge-name",
        action="store",
        default=name,
        help="Name of worker",
        metavar="STR",
    )
    group.add_argument(
        "--uge-n-cores",
        action="store",
        default=n_cores,
        help="Cores per worker",
        metavar="INT",
        type=int,
    )
    group.add_argument(
        "--uge-wait", action="store_true", help="Wait for UGE job to finish",
    )
    group.add_argument(
        "--uge-time",
        action="store",
        default=time,
        help="Max time for worker (e.g. format 30min, 1hour, 1day)",
        metavar="STR",
    )
    group.add_argument(
        "--uge-per-job",
        action="store",
        default=per_job,
        help="How many molecules per job",
        metavar="INT",
        type=int,
    )
    group.add_argument(
        "--uge-use-qrsh", action="store_true", help=argparse.SUPPRESS,
    )


def parse_time(timestr):
    """Parse human readable time"""

    if "h" in timestr or "hour" in timestr or "hours" in timestr:

        timestr = timestr.replace("h", "").replace("our", "").replace("s", "")
        timestr = int(timestr)
        hours = timestr
        mins = 0

    elif "m" in timestr or "min" in timestr:
        timestr = timestr.replace("m", "").replace("in", "").replace("s", "")
        timestr = int(timestr)

        hours = timestr // 60
        mins = timestr % 60

    elif "d" in timestr or "day" in timestr or "days" in timestr:
        timestr = timestr.replace("d", "").replace("ay", "").replace("s", "")
        timestr = int(timestr)

        hours = timestr * 24
        mins = 0

    else:
        _logger.error(f"I don't understand the time format {timestr}")
        return 0, 0

    return hours, mins


def get_chunks_df(df: DataFrame, n: int):
    """ Get chunks of dataframe, in chunks of n """

    num_chunks = len(df) // n + 1

    for i in range(num_chunks):
        yield df[i * n : (i + 1) * n]


def get_chunks_iter(lst: Iterable, n: int) -> Generator[Any, None, None]:
    """ Get the list in chunks of n """
    chunk = []
    N = 0

    for item in lst:

        if N == n:
            yield chunk
            N = 0
            chunk = []

        N += 1
        chunk.append(item)

    if len(chunk) > 0:
        yield chunk

    return


# Arg parsers
def parse_value(value: Any) -> Any:
    """ Parse value and make it command line friendly """

    if value is None:
        return None

    if isinstance(value, bool):
        return ""

    if isinstance(value, list):
        values = [parse_value(x) for x in value]
        return " ".join(values)

    if isinstance(value, str):
        if " " in value:
            value = f'"{value}"'
            return value

    return str(value)


def get_argument_string(options: dict) -> str:
    """ Generate arguments from options 

    e.g.
    {"name": "hello"}
    --name hello
    """

    line = []
    for key, val in options.items():

        # Ignore empty values
        if val is None:
            continue

        # Ignore False values
        if val is False:
            continue

        key = key.replace("_", "-")
        val_ = parse_value(val)
        cmd = f"--{key} {val_}"
        line.append(cmd)

    return " ".join(line)


def get_all_the_system_info():

    _logger.info("Reading current env and modules...")

    return SimpleNamespace(
        **dict(
            python_path=os.getenv("PYTHONPATH", None),
            conda_prefix=os.getenv("CONDA_PREFIX", None),
        )
    )
