"""
wurlitzer-lite using files instead of pipes

avoids max-pipe-size issue in exchange for dealing with files
"""

import ctypes
import os
import sys
from contextlib import contextmanager

# copied from wurlitzer
libc = ctypes.CDLL(None)
try:
    c_stdout_p = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr_p = ctypes.c_void_p.in_dll(libc, 'stderr')
except ValueError:
    # libc.stdout has a funny name on macOS
    c_stdout_p = ctypes.c_void_p.in_dll(libc, '__stdoutp')
    c_stderr_p = ctypes.c_void_p.in_dll(libc, '__stderrp')


@contextmanager
def capture_to_file(stdout="./stdout", stderr="./stderr"):
    stdout_f = stderr_f = None
    if stdout:
        stdout_f = open(stdout, mode="wb")
        real_stdout = sys.__stdout__.fileno()
        save_stdout = os.dup(real_stdout)
        os.dup2(stdout_f.fileno(), real_stdout)
    if stderr:
        stderr_f = open(stderr, mode="wb")
        real_stderr = sys.__stderr__.fileno()
        save_stderr = os.dup(real_stderr)
        os.dup2(stderr_f.fileno(), real_stderr)
    try:
        yield stdout, stderr
    finally:
        # flush to capture the last of it
        libc.fflush(c_stdout_p)
        libc.fflush(c_stderr_p)

        if stdout:
            os.dup2(save_stdout, real_stdout)
            stdout_f.close()
        if stderr:
            os.dup2(save_stderr, real_stderr)
            stderr_f.close()

# ======================================================================
# example

# calls via PyDLL hold the GIL,
# which blocks when using wurlitzer with pipes
# if the message size exceeds max-pipe-size
pylibc = ctypes.PyDLL(None)

def main():
    sz = 64000
    # while sz < 100_000_000:
    while sz < 100_000:
        print('writing', sz)
        buf = b'1' * sz

        with capture_to_file() as (stdout, stderr):
            # execute anything that outputs to stdout, stderr
            pylibc.printf(buf + b'\0')
        # gather from files
        stdoutstr, stderrstr = None, None
        with open(stdout) as outf, open(stderr) as errf:
            stdoutstr = outf.read()
            stderrstr = errf.read()
        # print strings
        print(stdoutstr)
        print(stderrstr)
        with capture_to_file() as (stdout, stderr):
            # execute anything that outputs to stdout, stderr
            pylibc.printf(buf + b'\0')
        # gather from files
        stdoutstr, stderrstr = None, None
        with open(stdout) as outf, open(stderr) as errf:
            stdoutstr = outf.read()
            stderrstr = errf.read()
        # print strings
        print(stdoutstr)
        print(stderrstr)


if __name__ == "__main__":
    main()
