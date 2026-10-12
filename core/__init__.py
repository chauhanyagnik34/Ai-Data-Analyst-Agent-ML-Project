"""Dataset loading, profiling, charts, and the optional intent classifier."""

from core.data_manager import (
    load_dataset,
    get_dataset_overview,
    generate_full_profile,
    get_column_types,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from core.visualization import create_auto_chart

__all__ = [
    "load_dataset",
    "get_dataset_overview",
    "generate_full_profile",
    "get_column_types",
    "MAX_FILE_SIZE_BYTES",
    "SUPPORTED_EXTENSIONS",
    "create_auto_chart",
]
