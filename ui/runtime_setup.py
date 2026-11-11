import os
import sys

from core.config import AgentConfig
from core.logging_config import setup_logging
from core.std_streams import ensure_standard_streams


def setup_runtime() -> AgentConfig:
    # Windowed builds have no console; stdio MCP servers need a real stderr.
    ensure_standard_streams()
    if getattr(sys, "frozen", False):
        os.chdir(os.getcwd())
    config = AgentConfig()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        reasoning_debug_enabled=config.debug_reasoning_stream,
    )
    return config
