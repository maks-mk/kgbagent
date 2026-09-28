import errno
import os
import shutil
import stat
import tempfile
from functools import lru_cache
from pathlib import Path


IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt",
    "venv", ".venv", "env", ".env",
    "dist", "build", "out", "target",
    ".idea", ".vscode",
}

_KNOWN_TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss", ".less",
    ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".csv", ".log", ".sh", ".bat", ".ps1",
    ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".rb", ".php",
    ".sql", ".env", ".gitignore", ".dockerignore", ".editorconfig",
}
_KNOWN_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".wav", ".flac", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pyc", ".pyo", ".class", ".o", ".obj",
}


@lru_cache(maxsize=512)
def is_binary_path(path_str: str) -> bool:
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext in _KNOWN_TEXT_EXTS:
        return False
    if ext in _KNOWN_BINARY_EXTS:
        return True
    try:
        with open(path_str, "rb") as file_obj:
            return b"\x00" in file_obj.read(8192)
    except Exception:
        return True


def count_file_lines(path: Path) -> int:
    try:
        if path.stat().st_size == 0:
            return 0

        count = 0
        last_chunk = b""
        with open(path, "rb") as file_obj:
            while chunk := file_obj.read(65536):
                count += chunk.count(b"\n")
                last_chunk = chunk

        if last_chunk and not last_chunk.endswith(b"\n"):
            count += 1
        return count
    except Exception:
        return 0


def candidate_path_inputs(path_str: str) -> list[str]:
    normalized = str(path_str).strip()
    candidates: list[str] = []
    for candidate in (
        normalized,
        normalized.strip("\"'"),
        normalized.rstrip(",;").rstrip(),
        normalized.strip("\"'").rstrip(",;").rstrip(),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_path(
    cwd: Path,
    virtual_mode: bool,
    path_str: str,
    extra_read_roots: tuple[Path, ...] = (),
) -> Path:
    if not path_str:
        raise ValueError("Path cannot be empty")
    if "\0" in str(path_str):
        raise ValueError("Path cannot contain NUL.")

    resolved_candidates: list[tuple[str, Path]] = []
    for candidate_input in candidate_path_inputs(path_str):
        clean_path = candidate_input.replace("\\", "/")
        path_obj = Path(clean_path).expanduser()

        if virtual_mode:
            # First, check if it's already an absolute path
            is_abs = path_obj.is_absolute()
            
            # Resolve the full path
            full_path = path_obj.resolve() if is_abs else (cwd / path_obj).resolve()
            
            # Check if it's relative to cwd (this handles traversal attempts like ../../etc)
            if not full_path.is_relative_to(cwd):
                # Read-only roots (e.g. the bundled skills folder) may live outside
                # the workspace. Allow paths that resolve inside one of them so the
                # model can read a skill's auxiliary files; traversal is still bounded
                # because full_path is fully resolved before this check.
                if any(full_path.is_relative_to(root) for root in extra_read_roots):
                    pass
                # Only allow absolute paths if they actually point inside the cwd
                elif is_abs:
                    raise ValueError(f"ACCESS DENIED: Absolute paths not allowed in virtual mode: {path_str}")
                else:
                    raise ValueError(f"ACCESS DENIED: Path traversal outside working directory: {full_path}")
        else:
            full_path = path_obj.resolve() if path_obj.is_absolute() else (cwd / path_obj).resolve()

        resolved_candidates.append((candidate_input, full_path))

    _, original_path = resolved_candidates[0]
    if original_path.exists():
        return original_path

    for _, candidate_path in resolved_candidates[1:]:
        if candidate_path.exists():
            return candidate_path

    return original_path


def resolve_existing_path(cwd: Path, virtual_mode: bool, path: str, expected: str, *, follow_final_symlink: bool = True) -> Path:
    # When the caller must not follow the final component (e.g. deleting a
    # symlink), resolve only the parent so is_symlink() still sees the link.
    # A full .resolve() would collapse it to the target and defeat the check.
    if not follow_final_symlink:
        literal = _resolve_without_final_symlink(cwd, virtual_mode, path)
        if literal.is_symlink():
            if expected == "file":
                return literal
            raise NotADirectoryError(f"{path} is a symlink. Use safe_delete_file.")
        target = literal
    else:
        target = resolve_path(cwd, virtual_mode, path)
    if not target.exists():
        raise FileNotFoundError(path)
    if expected == "file" and not target.is_file():
        raise IsADirectoryError(path)
    if expected == "dir" and not target.is_dir():
        raise NotADirectoryError(path)
    return target


def _resolve_without_final_symlink(cwd: Path, virtual_mode: bool, path: str) -> Path:
    """Resolve the parent chain but keep the final path component literal.

    resolve_path() fully resolves symlinks, collapsing a link to its target;
    that makes is_symlink() useless for delete operations. This rebuilds the
    path from the raw input (same candidate normalization as resolve_path)
    and resolves only the parent directory.
    """
    for candidate_input in candidate_path_inputs(path):
        clean_path = candidate_input.replace("\\", "/")
        path_obj = Path(clean_path).expanduser()
        if path_obj.is_absolute():
            absolute = path_obj
        else:
            absolute = cwd / path_obj
        if absolute.name in {"", ".", ".."}:
            return absolute.resolve()
        literal = absolute.parent.resolve() / absolute.name
        if virtual_mode and not literal.is_relative_to(cwd):
            raise ValueError(f"ACCESS DENIED: Path outside working directory: {path}")
        if literal.exists() or literal.is_symlink():
            return literal
    return resolve_path(cwd, virtual_mode, path)


def atomic_write_text(target: Path, content: str, *, expected_content: bytes | None = None) -> None:
    """Atomic UTF-8 replacement preserving permissions and caller-supplied newlines.

    The optional stale-content check is not a multi-process lock. Callers must
    still serialize overlapping writes; compare-and-replace is not a transaction.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            os.chmod(temp, stat.S_IMODE(target.stat().st_mode))
        if expected_content is not None and target.read_bytes() != expected_content:
            raise ValueError("File changed while preparing the edit; read it again before retrying.")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def delete_directory_path(target: Path, recursive: bool) -> None:
    if target.is_symlink():
        # rmtree on a symlink follows the link and deletes the target tree.
        raise ValueError("Use safe_delete_file to unlink a directory symlink.")
    if recursive:
        shutil.rmtree(target)
        return
    try:
        next(target.iterdir())
    except StopIteration:
        pass
    else:
        raise OSError(errno.ENOTEMPTY, "Directory is not empty.")
    target.rmdir()
