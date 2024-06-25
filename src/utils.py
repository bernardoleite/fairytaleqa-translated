from datetime import datetime
import re
import sys

def currentdate():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Find string between two substrings
# https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
def find_string_between_two_substring(complete_string, first_delimiter, second_delimiter):
    if second_delimiter == '<END>':
        regex = first_delimiter + '(.*)'
        result = re.search(regex, complete_string)
        return result.group(1).strip()
    else:
        regex = first_delimiter + '(.*)' + second_delimiter
        result = re.search(regex, complete_string)
        return result.group(1).strip()
