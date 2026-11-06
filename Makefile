.PHONY: dev dev-backend dev-frontend test test-backend build demo results clean

PYTHON = .venv/bin/python

dev-backend:
	$(PYTHON) -m uvicorn gridlab.api.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals"

test: test-backend

test-backend:
	$(PYTHON) -m pytest backend/tests/ -v

build:
	cd frontend && npm run build

demo: build
	$(PYTHON) -m uvicorn gridlab.api.main:app --port 8000

EXPERIMENTS = e1_cliff_contrast e2_alpha_sensitivity e3_exploration_strategy \
              e4_map_difficulty e4b_maze e4c_two_goals e5_adaptation e6_slippery

results: results-run results-plot

results-run:
	@mkdir -p results/runs results/figs
	@for exp in $(EXPERIMENTS); do \
		echo "Running $$exp..."; \
		$(PYTHON) -m gridlab.cli.run_experiment experiments/$$exp.yaml --seeds 20 --output results/runs/$$exp; \
	done

results-plot:
	@mkdir -p results/figs
	$(PYTHON) -m gridlab.cli.plot_results $(wildcard results/runs/*) --output results/figs

clean:
	rm -rf results/runs/* results/figs/*
	rm -rf frontend/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
