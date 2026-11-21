r"""
coPPeR desktop application package — editor bridge, inference server, and cursor overlay.
"""

__all__ = [
    "apply_selection_replacement",
    "capture_highlighted_selection",
    "detect_installed_editors",
    "get_active_window_context",
]

from .editor_bridge import (
    apply_selection_replacement as apply_selection_replacement,
)
from .editor_bridge import (
    capture_highlighted_selection as capture_highlighted_selection,
)
from .editor_bridge import (
    detect_installed_editors as detect_installed_editors,
)
from .editor_bridge import (
    get_active_window_context as get_active_window_context,
)
