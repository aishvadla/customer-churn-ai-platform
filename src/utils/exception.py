"""Custom exception handling for the ML project.

This module defines exception classes and utilities to capture detailed
error information including file names and line numbers for debugging.
"""

import sys


def error_message_details(error, error_detail: sys) -> str:
    """Extract detailed error information from exception traceback.

    Args:
        error: The exception object.
        error_detail: sys module to access exception info.

    Returns:
        str: Formatted error message with file name, line number, and error text.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    """Custom exception with detailed traceback information.

    Extends the standard Exception class to capture file names, line numbers,
    and original error messages for better debugging.
    """

    def __init__(self, error_message, error_detail: sys):
        """Initialize the custom exception.

        Args:
            error_message: The original error message or exception.
            error_detail: sys module to extract traceback info.
        """
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self) -> str:
        """Return the detailed error message including traceback info.

        Returns:
            str: Formatted error message with context.
        """
        return self.error_message
