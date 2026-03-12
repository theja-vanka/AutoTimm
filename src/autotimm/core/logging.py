"""Central logging configuration for AutoTimm using loguru."""

from __future__ import annotations

import sys

from loguru import logger

# Remove default handler and add a custom one with colorful format
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)


def log_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Log a formatted table using loguru.

    Args:
        title: Table title.
        headers: Column header names.
        rows: List of rows, each row is a list of string values.
    """
    if not rows:
        logger.info(f"{title}: (empty)")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build formatted table
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-" * len(header_line)

    lines = [title, separator, header_line, separator]
    for row in rows:
        line = "  ".join(
            str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell)
            for i, cell in enumerate(row)
        )
        lines.append(line)
    lines.append(separator)

    logger.info("\n".join(lines))


__all__ = ["logger", "log_table"]
