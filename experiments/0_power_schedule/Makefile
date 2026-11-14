PYTHON ?= uv run --frozen python
SEEDS ?= 0 1 2
PARTS = $(SEEDS:%=outputs/power_s%.json)

.PHONY: all clean
all: figs/power.png

outputs/power_s%.json: scripts/run.py | outputs
	OMP_NUM_THREADS=2 $(PYTHON) $< --seed $* --output $@

outputs/power.json: scripts/merge.py $(PARTS)
	$(PYTHON) $< --inputs $(PARTS) --output $@

figs/power.png: scripts/plot.py outputs/power.json | figs
	$(PYTHON) $< --input outputs/power.json --output $@

outputs figs:
	mkdir -p $@

clean:
	rm -f outputs/power*.json figs/power.png
