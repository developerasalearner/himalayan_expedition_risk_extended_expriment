.PHONY: help setup smoke full figure verify trap causes clean

help:
	@echo "setup   - create .venv and install dependencies"
	@echo "smoke   - 10-minute run that validates the environment"
	@echo "full    - the paper's run (~3.3 h on an M2 laptop)"
	@echo "figure  - rebuild the dataset overview figure with verified coordinates"
	@echo "verify  - re-derive the paper's conclusions from the stored tables"
	@echo "trap    - supplementary Table 4, calibration measures under the fault"
	@echo "causes  - death-date and cause-of-death counts used in Section 8.1"
	@echo ""
	@echo "macOS: run 'brew install libomp' first or XGBoost will not import."

setup:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	@echo "activate with: source .venv/bin/activate"

smoke:
	python scripts/master_experiment_v5.py --tier smoke --outdir smoke_results

full:
	python scripts/master_experiment_v5.py --tier full --outdir results

figure:
	python scripts/regenerate_figure1.py \
	  --coords results/artifacts/peak_coordinates_verified_final.csv \
	  --outdir results

verify:
	python scripts/verify_results.py --new results --reference results

trap:
	python scripts/trap_calibration_table.py

causes:
	python scripts/verify_death_causes.py

clean:
	rm -rf smoke_results _trap_tmp __pycache__ scripts/__pycache__
