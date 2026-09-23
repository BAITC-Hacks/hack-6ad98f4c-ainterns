"""Convenience entry point for the Streamlit dashboard."""
import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).resolve().parent / "src" / "app.py"), run_name="__main__")
