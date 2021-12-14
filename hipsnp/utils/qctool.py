import os
import subprocess
from pathlib import Path
from . logging import raise_error


def get_qctool():
    # Check if qctool is in path
    result = subprocess.run(
        ['qctool'], shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return 'qctool'

    # Check for QCTOOL_PATH environment variable
    qctool = os.environ.get('QCTOOL_PATH', None)
    if qctool is None:
        raise_error('qctool not found in path or QCTOOL_PATH not set')

    if not Path(qctool).is_file():
        raise_error(f'QCTOOL_PATH has a wrong value: {qctool}')
    result = subprocess.run(
        [qctool], shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return qctool

    raise_error('The executable in QCTOOL_PATH is not qctool or cannot be run')
