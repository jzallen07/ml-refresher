.PHONY: book jupyter serve stop help

# Start the book UI only (read-only, no code execution)
book:
	uv run jupyter-book start

# Start Jupyter kernel server in background (needed for code execution)
jupyter:
	@echo "Starting Jupyter server on http://localhost:8888..."
	@uv run jupyter lab --no-browser --NotebookApp.token=mlrefresher --NotebookApp.allow_origin='*' > /dev/null 2>&1 &
	@sleep 2
	@echo "Jupyter server running in background (token: mlrefresher)"

# Start both servers for live code execution
# 1. Starts Jupyter in background
# 2. Starts book UI in foreground
serve: jupyter
	@echo ""
	@echo "Starting book UI..."
	@echo "Click the power button on notebook pages to enable code execution"
	@echo ""
	uv run jupyter-book start

# Stop all running servers
stop:
	@pkill -f "jupyter-book" 2>/dev/null || true
	@pkill -f "jupyter-lab" 2>/dev/null || true
	@pkill -f "jupyter lab" 2>/dev/null || true
	@pkill -f "myst" 2>/dev/null || true
	@echo "All servers stopped"

# Show help
help:
	@echo "ML Refresher - Interactive Book Commands"
	@echo ""
	@echo "  make book    - Start book UI (read-only, no code execution)"
	@echo "  make serve   - Start book with live code execution"
	@echo "  make stop    - Stop all servers"
	@echo ""
	@echo "For code execution: run 'make serve', then click the power"
	@echo "button on any notebook page to connect to the Jupyter kernel."
