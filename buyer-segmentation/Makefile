.PHONY: pipeline dashboard test clean

pipeline:
	python src/pipeline.py

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

clean:
	rm -f data/processed/*
	rm -f outputs/models/*
	rm -f outputs/profiles/*
	rm -f outputs/figures/*
