#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

import launch
import pkg_resources

DEVICE_INFO_PATH = Path(__file__).absolute().parent / "last_device.txt"
_REQUIREMENT_PATH = Path(__file__).absolute().parent / "requirements.txt"


def _get_comparable_version(version: str) -> tuple:
    return tuple(version.split("."))


def _get_installed_version(package: str) -> str | None:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])


def set_device(value):
	with open(DEVICE_INFO_PATH, "w") as txt:
		txt.write(value)


import torch.cuda as cuda
if cuda.is_available():
	set_device('cuda')
	print(f'[ FACE_FUSION ] cuda available')
	pip_uninstall("onnxruntime", "onnxruntime-gpu")
	launch.run_pip('install -U "onnxruntime-gpu"')
else:
	set_device('cpu')
	print(f'[ FACE_FUSION ] use cpu')
	pip_uninstall("onnxruntime", "onnxruntime-gpu")
	launch.run_pip('install -U "onnxruntime"')


with _REQUIREMENT_PATH.open() as fp:
    for requirement in fp:
        try:
            requirement = requirement.strip()
            if "==" in requirement:
                name, version = requirement.split("==", 1)
                installed_version = _get_installed_version(name)

                if installed_version == version:
                    continue

                launch.run_pip(
                    f'install -U "{requirement}"',
                    f"sd-webui-facefusion requirement: changing {name} version from {installed_version} to {version}",
                )
                continue

            if ">=" in requirement:
                name, version = requirement.split(">=", 1)
                installed_version = _get_installed_version(name)

                if installed_version and (
                    _get_comparable_version(installed_version) >= _get_comparable_version(version)
                ):
                    continue

                launch.run_pip(
                    f'install -U "{requirement}"',
                    f"sd-webui-facefusion requirement: changing {name} version from {installed_version} to {version}",
                )
                continue

            if not launch.is_installed(requirement):
                launch.run_pip(
                    f'install "{requirement}"',
                    f"sd-webui-facefusion requirement: {requirement}",
                )
        except Exception as error:
            print(error)
            print(f"Warning: Failed to install '{requirement}', some preprocessors may not work.")
