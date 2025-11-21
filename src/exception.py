# For exception handling throughout the project.

import sys
import logging

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    # This line of code gets us where the errror occured, which file and which line number. 

    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error occured in python script name: [{0}], at line number: [{1}], error message: [{2}]'.format(
        file_name, # 0
        exc_tb.tb_lineno, # 1
        str(error) # 2
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
               super().__init__(error_message)
               self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
          return self.error_message