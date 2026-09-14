# Convenience commands for local development.

.PHONY: install train visualize dsp

install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

visualize:
	python -m src.visualize

visualize_dsp:
	python -m src.visualize_dsp

animate:
	python -m src.animate
