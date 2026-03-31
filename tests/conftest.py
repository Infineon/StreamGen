"""🗃️ fixtures available in all tests."""

import matplotlib

# Use a non-interactive backend in tests to avoid GUI/Tk dependencies.
matplotlib.use("Agg", force=True)
