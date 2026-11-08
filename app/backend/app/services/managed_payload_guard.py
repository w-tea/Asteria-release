"""Read sharing for a parked Windows runtime tree during replacement."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import shutil


class PayloadReadPin:
    def __init__(self, handle=None, *, close=None, dispose=None, reader=None, path=None):
        self.handle, self._close, self._dispose = handle, close, dispose
        self._reader, self.path = reader, path

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self._close(handle)

    def delete_original(self):
        if self.handle is None or self._dispose is None:
            raise OSError("handle-directed payload cleanup is unavailable on this host")
        self._dispose(self.handle)
        self.close()

    @contextmanager
    def reader(self):
        with (self._reader(self.handle) if self._reader else self.path.open("rb")) as stream:
            yield stream

    def sha256(self):
        digest = sha256()
        with self.reader() as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def copy_preserved_file(source: Path, target: Path, pin: PayloadReadPin):
    with pin.reader() as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    shutil.copystat(source, target)


@contextmanager
def hold_payload_read(path: Path, *, directory: bool = False):
    if os.name != "nt":
        # Namespace parking and inventory checks still apply on other hosts.
        yield PayloadReadPin(path=path)
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    disposition = kernel.SetFileInformationByHandle
    disposition.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    disposition.restype = wintypes.BOOL
    duplicate = kernel.DuplicateHandle
    duplicate.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                          ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    duplicate.restype = wintypes.BOOL
    current_process = kernel.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = wintypes.HANDLE
    flags = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT / BACKUP_SEMANTICS
    handle = create(str(path), 0x80000000 | 0x10000, 0x1 | 0x4, None, 3, flags, None)
    # READ | DELETE sharing permits the publisher's rename/removal, not writes.
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    def close_checked(value):
        if not close(value):
            raise ctypes.WinError(ctypes.get_last_error())

    def delete_opened_object(value):
        # FILE_DISPOSITION_INFO contains BOOLEAN (one byte), not Win32 BOOL.
        delete_file = ctypes.c_ubyte(1)
        if not disposition(value, 4, ctypes.byref(delete_file), ctypes.sizeof(delete_file)):
            raise ctypes.WinError(ctypes.get_last_error())

    @contextmanager
    def reader(value):
        duplicated = wintypes.HANDLE()
        process = current_process()
        if not duplicate(process, value, process, ctypes.byref(duplicated), 0, False, 2):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            descriptor = msvcrt.open_osfhandle(duplicated.value, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            close_checked(duplicated)
            raise
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.seek(0)
            yield stream

    pin = PayloadReadPin(handle, close=close_checked, dispose=delete_opened_object, reader=reader, path=path)
    try:
        yield pin
    finally:
        pin.close()
