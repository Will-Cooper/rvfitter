"""Entry point for the frozen rvfitter GUI (PyInstaller)."""
import multiprocessing
import sys

from rvfitter.gui.app import main

if __name__ == '__main__':
    multiprocessing.freeze_support()  # worker processes (forward_mcmc pool) re-enter the frozen executable
    sys.exit(main())
