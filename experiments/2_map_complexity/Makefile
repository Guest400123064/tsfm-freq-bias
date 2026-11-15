PYTHON ?= uv run --frozen python
SEEDS ?= 0 1 2
PARTS = $(SEEDS:%=outputs/map_s%.json)

.PHONY: all data smoke clean
all: outputs/map.json
data: outputs/data_pilot.json figs/data_pilot.png
smoke: outputs/smoke.json

outputs/data_pilot.json: scripts/diagnostics.py | outputs figs
	$(PYTHON) $< --output outputs/data_pilot.json --fig figs/data_pilot.png

figs/data_pilot.png: outputs/data_pilot.json ;

outputs/map_s%.json: scripts/run.py | outputs
	OMP_NUM_THREADS=2 $(PYTHON) $< --seed $* --output $@

outputs/map.json: scripts/merge.py $(PARTS)
	$(PYTHON) $< --inputs $(PARTS) --output $@

outputs/smoke.json: scripts/run.py | outputs
	OMP_NUM_THREADS=2 $(PYTHON) $< --seed 0 --steps 256 --output $@

outputs figs:
	mkdir -p $@

clean:
	rm -f outputs/*.json figs/*.png
