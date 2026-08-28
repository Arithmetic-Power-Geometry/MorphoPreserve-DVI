.PHONY: install test reproduce app clean
install:
	python -m pip install -e . pytest

test:
	pytest -q

reproduce:
	morphopreserve --data data/ranchi_nasal_morphometry.csv --out results --repeats 10 --splits 10 --bootstrap 500

app:
	streamlit run app.py

clean:
	rm -rf results .pytest_cache
