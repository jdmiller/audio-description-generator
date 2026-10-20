#!/bin/bash
# Suppresses macOS malloc subsystem noise (MallocStackLogging, MallocNanoZone)
# that is written directly to fd 2 at C library load time, before Python starts.
# Progress bars and status output use stdout; errors go to the run's log file.
exec env MallocStackLogging=0 MallocNanoZone=0 \
    python3 "$(dirname "$0")/describer.py" "$@" 2>/dev/null
