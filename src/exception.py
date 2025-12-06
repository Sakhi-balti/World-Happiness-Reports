import sys
from src.logger import logging
  

def get_error_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    message = (
        f"Error occurred in script: [{file_name}]\n"
        f"At line number: [{line_number}]\n"
        f"With error message: {error}\n"
    )
    return message


class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(error)
        self.error = get_error_detail(error, error_detail)

    def __str__(self):
        return self.error


