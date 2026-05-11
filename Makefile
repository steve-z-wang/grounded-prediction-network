.PHONY: test paper

test:
	PYTHONPATH=. python -m pytest tests/ -v

paper:
	$(MAKE) -C paper/v1-arxiv paper
