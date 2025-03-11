
from rich.console import Console
from rich.panel import Panel

console = Console()
import config as config


def styled_print(icon: str, text: str, color: str, bold: bool = True, panel: bool = False):
    """Prints formatted text with an icon, color, and optional bold styling inside a panel."""
    style = f"bold {color}" if bold else color
    message = f"{icon} [{style}]{text}[/{style}]"
    
    if panel:
        console.print(Panel(message, expand=False, border_style=color))
    else:
        console.print(message)