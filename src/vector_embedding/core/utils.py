"""
Utility functions for the vector_embedding package.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root by walking up the directory tree until config.toml is found.

    This function is useful for locating the project root from any module,
    enabling relative path resolution for configuration files and data directories.

    Returns:
        Path: Absolute path to the project root directory

    Raises:
        FileNotFoundError: If config.toml is not found in any parent directory

    Example:
        >>> root = get_project_root()
        >>> config_path = root / "config.toml"
    """
    current = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / "config.toml").exists():
            return parent

    # Fallback: if not found, raise error
    raise FileNotFoundError(
        "Could not find project root (config.toml not found). "
        f"Started search from: {Path(__file__)}"
    )
