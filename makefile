format:
	black .
	ruff check .
	ruff format .

lint:
	ruff check .

test:
	pytest test/test_current_system.py

run:
	python chat.py