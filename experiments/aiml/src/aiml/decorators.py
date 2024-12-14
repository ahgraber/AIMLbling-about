# %%
import datetime as dt
from functools import partial, wraps
import inspect
import logging
import time

# %%
logger = logging.getLogger(__name__)


# %%
# shamelessly plagarized from scikit-lego: https://github.com/koaning/scikit-lego/blob/main/sklego/pandas_utils.py
def runtimer(func=None, *, print_fn=print, record_start=True):
    """Time a function using decorator and record success/failure.

    Note that once applied to the function, it will always be applied until function is redefined.
    To run function without wrapper, run `<funcname>.__wrapped__`.

    Examples
    --------
    >>> @runtimer
    >>> def do_something(x):
    >>>     for _ in range(x):
    >>>        sleep (1)
    >>>        return

    >>> @runtimer(print_fn=logging.info)
    >>> def do_something(x):
    >>>     for _ in range(x):
    >>>        sleep (1)
    >>>        return
    """
    if func is None:
        return partial(
            runtimer,
            print_fn=print_fn,
            record_start=record_start,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = dt.datetime.now()
        if record_start:
            print_fn(f"Starting '{func.__name__}' at {start_time}...")

        try:
            result = func(*args, **kwargs)
            run_time = dt.datetime.now() - start_time
            print_fn(f"'{func.__name__}' succeeded in: {run_time}")
        except Exception as err:
            print_fn(f"'{func.__name__}' failed")
            raise err
        else:
            return result

    return wrapper


# %%
