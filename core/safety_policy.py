from pydantic import BaseModel, Field
from core.config import DEFAULT_MAX_FILE_SIZE, DEFAULT_READ_LIMIT

class SafetyPolicy(BaseModel):
    """
    Centralized configuration for safety limits and policies.
    """
    max_tool_output: int = Field(default=5000, description="Maximum characters for tool output")
    max_raw_tool_output: int = Field(default=100000, description="Maximum characters captured before tool output compression")
    max_file_size: int = Field(default=DEFAULT_MAX_FILE_SIZE, description="Maximum file size in bytes")
    max_background_processes: int = Field(default=5, description="Maximum concurrent background processes")
    max_search_chars: int = Field(default=10000, description="Maximum characters for search results")
    max_read_lines: int = Field(default=DEFAULT_READ_LIMIT, description="Maximum lines to read from a file")
    allow_shell: bool = Field(default=False, description="Allow execution of shell commands")
