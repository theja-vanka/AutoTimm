"""Shared test configuration."""

import matplotlib

# Use non-interactive backend so tests work in headless CI (no display/Tk).
matplotlib.use("Agg")
