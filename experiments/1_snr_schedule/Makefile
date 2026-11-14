PYTHON ?= uv run --frozen python
SEEDS ?= 0 1 2
SNR_PARTS = $(SEEDS:%=outputs/snr_s%.json)
ASSIGN_PARTS = $(SEEDS:%=outputs/assignment_s%.json)

.PHONY: all snr assignment clean
all: outputs/snr.json outputs/assignment.json
snr: outputs/snr.json
assignment: outputs/assignment.json

outputs/snr_s%.json: scripts/run.py | outputs
	OMP_NUM_THREADS=2 $(PYTHON) $< --suite snr --seed $* --output $@

outputs/assignment_s%.json: scripts/run.py | outputs
	OMP_NUM_THREADS=2 $(PYTHON) $< --suite assignment --seed $* --output $@

outputs/snr.json: scripts/merge.py $(SNR_PARTS)
	$(PYTHON) $< --inputs $(SNR_PARTS) --output $@

outputs/assignment.json: scripts/merge.py $(ASSIGN_PARTS)
	$(PYTHON) $< --inputs $(ASSIGN_PARTS) --output $@

outputs:
	mkdir -p $@

clean:
	rm -f outputs/snr*.json outputs/assignment*.json
