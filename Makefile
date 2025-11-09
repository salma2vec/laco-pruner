.PHONY: install test toy

install:
	pip install -r requirements.txt

test:
	pytest -q

toy:
	bash run_toy_pipeline.sh

