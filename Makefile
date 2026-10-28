PY ?= .venv/bin/python
CSV ?= m1_jyot/healthcare-dataset-stroke-data.csv

.PHONY: all setup data train explain docs test app clean

all: data train explain docs test

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) make_data.py $(CSV)

train:
	$(PY) train.py

explain:
	$(PY) explain.py

docs:
	$(PY) generate_docs.py

test:
	$(PY) -m pytest tests/ -v

app:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf models reports/figures reports/*.csv reports/metrics.json reports/results.md
