"""
Functions to parse datetime objects.

We're using regular expressions rather than time.strptime because:
- They provide both validation and parsing.
- They're more flexible for datetimes.
- The date/datetime/time constructors produce friendlier error messages.

Stolen from https://raw.githubusercontent.com/django/django/main/django/utils/dateparse.py at
9718fa2e8abe430c3526a9278dd976443d4ae3c6

Changed to:
* use standard python datetime types not django.utils.timezone
* raise ValueError when regex doesn't match rather than returning None
* support parsing unix timestamps for dates and datetimes
"""
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Optional, Type, Union
from pydantic.v1 import errors
date_expr = '(?P<year>\\d{4})-(?P<month>\\d{1,2})-(?P<day>\\d{1,2})'
time_expr = '(?P<hour>\\d{1,2}):(?P<minute>\\d{1,2})(?::(?P<second>\\d{1,2})(?:\\.(?P<microsecond>\\d{1,6})\\d{0,6})?)?(?P<tzinfo>Z|[+-]\\d{2}(?::?\\d{2})?)?$'
date_re = re.compile(f'{date_expr}$')
time_re = re.compile(time_expr)
datetime_re = re.compile(f'{date_expr}[T ]{time_expr}')
standard_duration_re = re.compile('^(?:(?P<days>-?\\d+) (days?, )?)?((?:(?P<hours>-?\\d+):)(?=\\d+:\\d+))?(?:(?P<minutes>-?\\d+):)?(?P<seconds>-?\\d+)(?:\\.(?P<microseconds>\\d{1,6})\\d{0,6})?$')
iso8601_duration_re = re.compile('^(?P<sign>[-+]?)P(?:(?P<days>\\d+(.\\d+)?)D)?(?:T(?:(?P<hours>\\d+(.\\d+)?)H)?(?:(?P<minutes>\\d+(.\\d+)?)M)?(?:(?P<seconds>\\d+(.\\d+)?)S)?)?$')
EPOCH = datetime(1970, 1, 1)
MS_WATERSHED = int(20000000000.0)
MAX_NUMBER = int(3e+20)
StrBytesIntFloat = Union[str, bytes, int, float]

def parse_date(value: Union[date, StrBytesIntFloat]) -> date:
    """
    Parse a date/int/float/string and return a datetime.date.

    Raise ValueError if the input is well formatted but not a valid date.
    Raise ValueError if the input isn't well formatted.
    """
    if isinstance(value, date):
        return value
    elif isinstance(value, (int, float)):
        return (EPOCH + timedelta(seconds=int(value))).date()
    elif isinstance(value, bytes):
        value = value.decode()
    
    match = date_re.match(value)
    if match:
        kw = {k: int(v) for k, v in match.groupdict().items()}
        return date(**kw)
    else:
        raise ValueError('invalid date format')

def parse_time(value: Union[time, StrBytesIntFloat]) -> time:
    """
    Parse a time/string and return a datetime.time.

    Raise ValueError if the input is well formatted but not a valid time.
    Raise ValueError if the input isn't well formatted, in particular if it contains an offset.
    """
    if isinstance(value, time):
        return value
    elif isinstance(value, bytes):
        value = value.decode()
    
    match = time_re.match(value)
    if match:
        kw = match.groupdict()
        if kw['microsecond']:
            kw['microsecond'] = kw['microsecond'].ljust(6, '0')
        kw = {k: int(v) if v else 0 for k, v in kw.items() if k != 'tzinfo'}
        if match.groupdict()['tzinfo']:
            raise ValueError('offset-aware times are not supported')
        return time(**kw)
    else:
        raise ValueError('invalid time format')

def parse_datetime(value: Union[datetime, StrBytesIntFloat]) -> datetime:
    """
    Parse a datetime/int/float/string and return a datetime.datetime.

    This function supports time zone offsets. When the input contains one,
    the output uses a timezone with a fixed offset from UTC.

    Raise ValueError if the input is well formatted but not a valid datetime.
    Raise ValueError if the input isn't well formatted.
    """
    if isinstance(value, datetime):
        return value
    elif isinstance(value, (int, float)):
        return EPOCH + timedelta(seconds=int(value))
    elif isinstance(value, bytes):
        value = value.decode()
    
    match = datetime_re.match(value)
    if match:
        kw = match.groupdict()
        if kw['microsecond']:
            kw['microsecond'] = kw['microsecond'].ljust(6, '0')
        tzinfo = kw.pop('tzinfo')
        kw = {k: int(v) if v else 0 for k, v in kw.items()}
        if tzinfo == 'Z':
            tzinfo = timezone.utc
        elif tzinfo:
            offset_mins = int(tzinfo[-2:]) if len(tzinfo) > 3 else 0
            offset = 60 * int(tzinfo[1:3]) + offset_mins
            if tzinfo[0] == '-':
                offset = -offset
            tzinfo = timezone(timedelta(minutes=offset))
        else:
            tzinfo = None
        return datetime(**kw, tzinfo=tzinfo)
    else:
        raise ValueError('invalid datetime format')

def parse_duration(value: StrBytesIntFloat) -> timedelta:
    """
    Parse a duration int/float/string and return a datetime.timedelta.

    The preferred format for durations in Django is '%d %H:%M:%S.%f'.

    Also supports ISO 8601 representation.
    """
    if isinstance(value, timedelta):
        return value
    elif isinstance(value, (int, float)):
        return timedelta(seconds=int(value))
    elif isinstance(value, bytes):
        value = value.decode()
    
    match = standard_duration_re.match(value)
    if match:
        kw = match.groupdict()
        days = float(kw.pop('days') or 0)
        sign = -1 if kw.pop('sign', '+') == '-' else 1
        if kw.get('microseconds'):
            kw['microseconds'] = kw['microseconds'].ljust(6, '0')
        kw = {k: float(v) for k, v in kw.items() if v is not None}
        return sign * timedelta(days=days, **kw)
    
    match = iso8601_duration_re.match(value)
    if match:
        kw = match.groupdict()
        sign = -1 if kw.pop('sign') == '-' else 1
        days = float(kw.pop('days') or 0)
        kw = {k: float(v) for k, v in kw.items() if v is not None}
        return sign * timedelta(days=days, **kw)
    
    raise ValueError('invalid duration format')
