"""Test script entry points without installing robot drivers."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "scripts"), str(PROJECT / "src")]
