install:
	rm -rf venv  # Remove the existing venv folder if it exists
	python3 -m venv venv  # Create a new virtual environment
	. venv/bin/activate && pip install --upgrade pip  # Upgrade pip
	. venv/bin/activate && pip install -r requirements.txt  # Install dependencies

run:
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000
