# coding=utf-8


import copy
import logging
import sys
from logging import getLogger, Logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


def init(log_level: int) -> None:
    logger = get_package_logger()
    logger.propagate = False

    # Add handler if we don't have one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(log_level)


def get_package_logger() -> Logger:
    return getLogger('facefusion')


def debug(message: str, scope: str = 'FaceFusion') -> None:
    get_package_logger().debug('[' + scope + '] ' + message)


def info(message: str, scope: str = 'FaceFusion') -> None:
    get_package_logger().info('[' + scope + '] ' + message)


def warn(message: str, scope: str = 'FaceFusion') -> None:
    get_package_logger().warning('[' + scope + '] ' + message)


def error(message: str, scope: str = 'FaceFusion') -> None:
    get_package_logger().error('[' + scope + '] ' + message)


def enable() -> None:
    get_package_logger().disabled = False


def disable() -> None:
    get_package_logger().disabled = True


init(logging.INFO)
