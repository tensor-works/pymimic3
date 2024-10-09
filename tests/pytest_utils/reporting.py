import xml.etree.ElementTree as ET
import sys
from colorama import Fore, Style, init
from typing import List, Dict, Any, Set
from collections import defaultdict

# Initialize colorama
init()


class TestNode:
    """
    A node in the test tree representing either a test or a test container.

    Parameters
    ----------
    name : str
        The name of the test or test container.
    """

    def __init__(self, name):
        self.name = name
        self.children = {}
        self.is_test = False
        self.error_message = None


def create_test_tree(failed_tests: List[Dict[str, Any]]) -> TestNode:
    """
    Create a tree structure from a list of failed tests.

    The function splits each test name by dots and creates a tree structure
    where each level corresponds to a part of the test name.
    
    Parameters
    ----------
    failed_tests : List[Dict[str, Any]]
        A list of dictionaries, each containing information about a failed test.
        Each dictionary must have 'name' and 'message' keys.

    Returns
    -------
    TestNode
        The root node of the created tree.
    """
    root = TestNode("")

    for test in failed_tests:
        current = root
        parts = test['name'].split('.')

        # Process all parts except the last one (test name)
        for part in parts[:-1]:
            if part not in current.children:
                current.children[part] = TestNode(part)
            current = current.children[part]

        # Process the last part (test name)
        last_part = parts[-1]
        if last_part not in current.children:
            current.children[last_part] = TestNode(last_part)
        current.children[last_part].is_test = True
        current.children[last_part].error_message = test['message']

    return root


def print_tree(node: TestNode,
               prefix: str = "",
               is_last: bool = True,
               colors: List[str] = None) -> None:
    """
    Print a visual representation of the test tree.

    Parameters
    ----------
    node : TestNode
        The current node in the tree to print.
    prefix : str, optional
        The prefix to use for this line (used for recursion), by default "".
    is_last : bool, optional
        Whether this node is the last child of its parent, by default True.
    colors : List[str], optional
        List of color codes to use for different tree levels, by default None.
        If None, uses [Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.YELLOW].
    """

    if colors is None:
        colors = [Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.YELLOW]

    if node.name:
        connector = "└── " if is_last else "├── "
        color = colors[len(prefix) % len(colors)]

        if node.is_test:
            print(f"{prefix}{connector}{color}{node.name}{Style.RESET_ALL}")
            print(
                f"{prefix}{'    ' if is_last else '│   '}{Fore.RED}✗ Error{Style.RESET_ALL}: {node.error_message}"
            )
            if is_last and prefix:  # Add extra newline after last test in a group
                # Print the vertical lines for the extra newline
                parent_prefix = prefix.rstrip()  # Remove trailing spaces
                if parent_prefix:
                    print(f"{parent_prefix}")
        else:
            print(f"{prefix}{connector}{color}{node.name}{Style.RESET_ALL}")

    children = list(node.children.values())
    for i, child in enumerate(children):
        is_last_child = i == len(children) - 1
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(child, new_prefix, is_last_child, colors)

        # Add extra newline after last non-test child if it has no children
        if is_last_child and not child.is_test and not child.children:
            parent_prefix = prefix.rstrip()  # Remove trailing spaces
            if parent_prefix:
                print(f"{parent_prefix}")


def parse_junit_xml(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a JUnit XML file and extract information about failed tests.

    Parameters
    ----------
    file_path : str
        The path to the JUnit XML file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing information about a failed test.
        Each dictionary has 'name' and 'message' keys.

    Raises
    ------
    SystemExit
        If the file cannot be found or parsed.
    """
    failed_tests = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for testcase in root.findall('.//testcase'):
            failure = testcase.find('failure')
            error = testcase.find('error')

            if failure is not None or error is not None:
                error_element = failure if failure is not None else error
                message = error_element.get('message', '')
                short_message = message.split('\n')[0][:100] + ('...' if len(message) > 100 else '')

                test_info = {
                    'name': f"{testcase.get('classname', '')}.{testcase.get('name', '')}",
                    'message': short_message
                }
                failed_tests.append(test_info)

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

    return failed_tests


def report_failed_tests(failed_tests: List[Dict[str, Any]]) -> int:
    """
    Print a visual tree representation of failed tests.

    Parameters
    ----------
    failed_tests : List[Dict[str, Any]]
        A list of dictionaries, each containing information about a failed test.
        Each dictionary must have 'name' and 'message' keys.

    Returns
    -------
    int
        0 if all tests passed, 1 if any tests failed.

    Notes
    -----
    This function creates a tree structure from the failed tests and prints it
    to the console in a visually appealing format with colors and connectors.
    """
    if not failed_tests:
        print(f"{Fore.GREEN}All tests passed successfully!{Style.RESET_ALL}")
        return 0

    print(f"{Fore.RED}Failed tests ({len(failed_tests)}):{Style.RESET_ALL}")
    root = create_test_tree(failed_tests)
    print_tree(root)
    return 1


if __name__ == "__main__":
    from pathlib import Path
    file_path = Path(sys.argv[1])
    failed_tests = parse_junit_xml(file_path)
    exit_code = report_failed_tests(failed_tests)
    sys.exit(exit_code)
