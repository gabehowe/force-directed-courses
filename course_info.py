#!/usr/bin/env python

"""
Rudimentary HTML scraper for looking up USC courses.

Usage: 
    course_info.py [-s] <course code>
Examples: 
    course_info.py MATH 300
    course_info.py -s MATH 554
"""


import re
import requests
import sys
from bs4 import BeautifulSoup

URL = "https://academicbulletins.sc.edu/ribbit/index.cgi?page=getcourse.rjs&code="


def main(args):
    if len(args) < 1:
        print("Not enough arguments!", file=sys.stderr)
        return
    args[0] = args[0].upper().strip()

    # specifically allow for a space before the number w/o quotes, i.e., MATH 300 rather than "MATH 300"
    if len(args) > 1:
        args[0] += ' '+args[1].upper().strip()
    req = requests.get(URL + args[0])
    soup = BeautifulSoup(req.content, 'xml')
    html = BeautifulSoup(soup.contents[0].contents[1].contents[0], 'lxml')
    title_bits = [i.text for i in html.find_all('strong')]
    title = title_bits[1].strip().removeprefix('-').strip()
    credit_hours = re.match(r'.*(\d+).*', title_bits[2]).group(1)
    description = html.find(class_='courseblockextra').text.strip()
    prereqs = [i['title'].replace('\xa0', ' ') for i in html.find_all(class_='bubblelink')]
    try:
        gpa_code = html.find('strong', string="Carolina Core: ").parent.contents[1]
    except AttributeError:
        gpa_code = None
    return {
        'code': args[0],
        'title': title,
        'credits': credit_hours,
        'description': description,
        'prerequisites': prereqs,
        'gpa_code': gpa_code
    }

def sheetify(x):
    """Turn the data into a format acceptable for pasting into Google Sheets."""
    ci = main(x)
    print(ci['code'], ci['title'], ci['credits'], ','.join(ci['prerequisites']), sep=';')

if __name__ == '__main__':
    args: list = sys.argv[1:]
    try: 
        args.remove('-s')
        sheetify(args)
    except ValueError:
        print(main(args))



