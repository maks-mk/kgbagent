"""ssh-mcp package."""

from .windows_env import ensure_complete_environment

__all__ = ["__version__"]

__version__ = "0.3.1"

# The client that launches this stdio server may forward only a subset of its
# environment, dropping variables Windows OpenSSH requires to start (notably
# ProgramData, without which ssh.exe exits 255 with no output).  Repair the
# process environment here so that every child process spawned by this package
# inherits a complete one.
ensure_complete_environment()
