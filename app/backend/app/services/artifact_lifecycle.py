from functools import wraps
import threading


_condition = threading.Condition()
_active_writers = 0
_cleanup_active = False


def artifact_write_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _active_writers
        with _condition:
            while _cleanup_active:
                _condition.wait()
            _active_writers += 1
        try:
            return function(*args, **kwargs)
        finally:
            with _condition:
                _active_writers -= 1
                _condition.notify_all()

    return wrapped


def begin_orphan_cleanup() -> bool:
    global _cleanup_active
    with _condition:
        if _active_writers or _cleanup_active:
            return False
        _cleanup_active = True
        return True


def end_orphan_cleanup() -> None:
    global _cleanup_active
    with _condition:
        _cleanup_active = False
        _condition.notify_all()
