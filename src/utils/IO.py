"""Utility file

This file implements functionalities used by other modules and accessible to 
the user

"""

# Import necessary packages
import datetime
import os
import datetime
import inspect
import pdb
import sys
from pathlib import Path
from colorama import init, Fore, Style

__all__ = ["info_io", "info_io", "tests_io", "error_io", "debug_io", "warn_io", "suppress_io"]

WORKINGDIR = os.getenv("WORKINGDIR")

# Initialize Colorama
init(autoreset=True)


def _clean_path(path):
    """
    If path is inside of the directroy, make it relative to directory.
    """
    path = Path(path).relative_to(WORKINGDIR)
    if str(path).startswith("src"):
        return Path(path).relative_to("src")
    elif str(path).startswith("tests"):
        return Path(path).relative_to("tests")
    elif str(path).startswith("script"):
        return Path(path).relative_to("script")
    elif str(path).startswith("examples"):
        return Path(path).relative_to("examples")


DEBUG_OPTION = None
main_name = Path(sys.argv[0])
if "pytest" in str(main_name):
    PATH_PADDING = max([
        len(str(_clean_path(path)))
        for path in Path(WORKINGDIR).glob("**/*")
        if path.suffix == ".py"
    ])
else:
    # Doxygen may break on this
    try:
        PATH_PADDING = max([
            len(str(_clean_path(path)))
            for path in Path(WORKINGDIR).glob("**/*")
            if path.suffix == ".py" and not path.name.startswith("tests")
        ] + [len(str(main_name.relative_to(WORKINGDIR)))])
    except ValueError:
        PATH_PADDING = 0
# {path: len(str(path)) for path in Path().glob("**/*") if path.suffix == ".py"}
SRC_DIR = Path(WORKINGDIR, "src")
SCRIPT_DIR = Path(WORKINGDIR, "scripts")
TEST_DIR = Path(os.getenv("TESTS"))
EXAMPLE_DIR = Path(os.getenv("EXAMPLES"))
# Might need adjustment
HEADER_LENGTH = 5 + 19 + 3 + PATH_PADDING + 3 + 2 + 4
TAG_PADDING = 8


def _get_relative_path(file_path: Path):
    """
    Get the relative path based on the file location and predefined directories.
    """
    file_path = str(file_path)
    if "src" in file_path:
        relativa_path = SRC_DIR
        return Path(file_path).relative_to(relativa_path)
    elif "script" in file_path:
        relativa_path = SCRIPT_DIR
        return Path(file_path).relative_to(relativa_path)
    elif "tests" in file_path:
        relativa_path = TEST_DIR
        return Path(file_path).relative_to(relativa_path)
    elif "examples" in file_path:
        relativa_path = EXAMPLE_DIR
        return Path(file_path).relative_to(relativa_path)
    return file_path


def _get_line(caller):
    """
    Generates a formatted line with the caller's file path and line number.
    """
    path = str(_get_relative_path(caller.filename)) + " "
    return f"{Fore.LIGHTWHITE_EX}{str(datetime.datetime.now().strftime('%m-%d %H:%M:%S'))[:19]}{Style.RESET_ALL} : {path:-<{PATH_PADDING}} : {Fore.LIGHTCYAN_EX}L {caller.lineno:<4}{Style.RESET_ALL}"


def _print_iostring(string: str, line_header: str, flush_block: bool, collor: str):
    """
    Prints the message itself without the header.
    """
    if flush_block:
        lines = string.split('\n')
        num_lines = len(lines)
        sys.stdout.write(f"\x1b[{num_lines}A")
        sys.stdout.write('\x1b[2K')
        print(lines[0])
        for line in lines[1:]:
            sys.stdout.write('\x1b[2K')
            print(" " * HEADER_LENGTH + f"{collor}-{Style.RESET_ALL} {line}")

        sys.stdout.flush()
        return True
    elif "\n" in string:
        lines = string.split('\n')
        print(lines[0])
        for line in lines[1:]:
            print(" " * HEADER_LENGTH + f"{collor}-{Style.RESET_ALL} {line}")
        return True
    return False


def info_io(message: str,
            level: int = None,
            end: str = None,
            flush: bool = None,
            unflush: bool = None,
            flush_block: bool = False):
    """
    Prints a message with an information header, which can be formatted at different levels, or without any header formatting.
    Level 0: Wide header
    Level 1: Medium header
    Level 2: Narrow header
    Level None: No header formatting, behaves like the original info_io.

    Parameters
    ----------
    message : str
        The message to log.
    level : int, optional
        The level of the header, or None for no header formatting. Defaults to None.
    end : str, optional
        How to end the print, e.g., "\n" (newline). Defaults to None.
    flush : bool, optional
        Whether to forcibly flush the stream. Defaults to None.
    unflush : bool, optional
        If True, adds an additional newline before the message. Defaults to None.
    flush_block : bool, optional
        If True, prints the whole block at once to avoid disruptions. Defaults to False.
    """
    base_io("INFO", Fore.BLUE, message, level, end, flush, unflush, flush_block)


def tests_io(message: str,
             level: int = None,
             end: str = None,
             flush: bool = None,
             unflush: bool = None,
             flush_block: bool = False):
    """
    Prints a message with an information header, which can be formatted at different levels, or without any header formatting.
    Level 0: Wide header
    Level 1: Medium header
    Level 2: Narrow header
    Level None: No header formatting, behaves like the original info_io.

    Parameters
    ----------
    message : str
        The message to log.
    level : int, optional
        The level of the header, or None for no header formatting. Defaults to None.
    end : str, optional
        How to end the print, e.g., "\n" (newline). Defaults to None.
    flush : bool, optional
        Whether to forcibly flush the stream. Defaults to None.
    unflush : bool, optional
        If True, adds an additional newline before the message. Defaults to None.
    flush_block : bool, optional
        If True, prints the whole block at once to avoid disruptions. Defaults to False.
    """
    base_io("TEST", Fore.GREEN, message, level, end, flush, unflush, flush_block)


def debug_io(message: str,
             end: str = None,
             flush: bool = None,
             unflush: bool = None,
             flush_block: bool = False):
    """
    Prints a debug message if the DEBUG_OPTION is set to a non-zero value.

    Parameters
    ----------
    message : str
        The debug message to log.
    end : str, optional
        How to end the print, e.g., "\n" (newline). Defaults to None.
    flush : bool, optional
        Whether to forcibly flush the stream. Defaults to None.
    unflush : bool, optional
        If True, adds an additional newline before the message. Defaults to None.
    flush_block : bool, optional
        If True, prints the whole block at once to avoid disruptions. Defaults to False.
    """
    # TODO: Improve this solution (Adopted because file level declaration will initialize before the var can be changed by test fixtures)
    global DEBUG_OPTION
    if DEBUG_OPTION is None:
        DEBUG_OPTION = os.getenv("DEBUG", "0")
    if DEBUG_OPTION == "0":
        return
    base_io("DEBUG", Fore.YELLOW, message, None, end, flush, unflush, flush_block)


def warn_io(message: str,
            end: str = None,
            flush: bool = None,
            unflush: bool = None,
            flush_block: bool = False):
    """
    Prints a warning.

    Parameters
    ----------
    message : str
        The warning message to log.
    end : str, optional
        How to end the print, e.g., "\n" (newline). Defaults to None.
    flush : bool, optional
        Whether to forcibly flush the stream. Defaults to None.
    unflush : bool, optional
        If True, adds an additional newline before the message. Defaults to None.
    flush_block : bool, optional
        If True, prints the whole block at once to avoid disruptions. Defaults to False.
    """
    base_io("WARN", "\033[38;5;208m", message, None, end, flush, unflush, flush_block)


def error_io(message: str,
             exception_type: Exception,
             end: str = None,
             flush: bool = None,
             unflush: bool = None,
             flush_block: bool = False):
    """
    Prints an error message with an error header and raises the specified exception.

    Parameters
    ----------
    message : str
        The error message to log.
    exception_type : Exception
        The type of exception to raise after logging the message.
    end : str, optional
        How to end the print, e.g., "\n" (newline). Defaults to None.
    flush : bool, optional
        Whether to forcibly flush the stream. Defaults to None.
    unflush : bool, optional
        If True, adds an additional newline before the message. Defaults to None.
    flush_block : bool, optional
        If True, prints the whole block at once to avoid disruptions. Defaults to False.

    Raises
    ------
    exception_type
        The specified exception type is raised after the error message is logged.
    """
    base_io("ERROR", Fore.RED, message, None, end, flush, unflush, flush_block)
    raise exception_type


def base_io(info_tag: str,
            collor: str,
            message: str,
            level: int = None,
            end: str = None,
            flush: bool = None,
            unflush: bool = None,
            flush_block: bool = False):

    # Get caller information
    caller = inspect.getframeinfo(inspect.stack()[2][0])
    line_header = _get_line(caller)

    if level is None:
        header_message = message
    else:
        assert level in [0, 1, 2], ValueError("Header level must be one of 0, 1, 2")
        width = [60, 50, 40][level]
        header_message = '-' * 10 + ' ' + collor + f'{message}{Style.RESET_ALL} '.ljust(
            width + len({Style.RESET_ALL}), "-")
        flush_block = False  # Disable flush_block when level is specified
    collor_info_tag = f"{collor}{info_tag}{Style.RESET_ALL} "
    info_tag_padding = TAG_PADDING + len(collor) + len(Style.RESET_ALL)

    io_string = f"{collor_info_tag.ljust(info_tag_padding, '-')} {line_header} {collor}-{Style.RESET_ALL} {header_message}"

    if level in [0, 1]:
        print(" " * HEADER_LENGTH + "-")
    if unflush:
        print()
    if not _print_iostring(io_string, line_header, flush_block, collor):
        if flush:
            end = "\r"
        print(io_string, end=end, flush=flush)


#https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class suppress_io:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
