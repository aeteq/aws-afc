"""The nova_balanced stand-in with possession read correctly (a workshop team that fixed the bug)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nova_balanced import commands_fixed as commands  # noqa: E402,F401
