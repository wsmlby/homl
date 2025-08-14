
from pathlib import Path
import sys


def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource, handling running from source and from
    a PyInstaller bundle.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in a normal Python environment
        base_path = Path(__file__).parent.absolute()
    return base_path / relative_path