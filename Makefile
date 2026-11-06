.PHONY: dev dev-backend dev-frontend test test-backend build demo results clean

dev-backend:
	cd backend && python -m uvicorn gridlab.api.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals"

test: test-backend

test-backend:
	cd backend && python -m pytest tests/ -v

build:
	cd frontend && npm run build

demo: build
	cd backend && python -m uvicorn gridlab.api.main:app --port 8000

results:
	cd backend && python -m gridlab.cli.run_experiment experiments/e1_cliff.yaml --seeds 20
	cd backend && python -m gridlab.cli.run_experiment experiments/e2_alpha.yaml --seeds 20
	cd backend && python -m gridlab.cli.run_experiment experiments/e3_exploration.yaml --seeds 20
	cd backend && python -m gridlab.cli.run_experiment experiments/e4_difficulty.yaml --seeds 20
	cd backend && python -m gridlab.cli.run_experiment experiments/e5_adaptation.yaml --seeds 20
	cd backend && python -m gridlab.cli.run_experiment experiments/e6_slippery.yaml --seeds 20
	cd backend && python -m gridlab.cli.plot_results

clean:
	rm -rf results/runs/* results/csv/* results/figs/*
	rm -rf frontend/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
