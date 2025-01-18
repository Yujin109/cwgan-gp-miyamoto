.PHONY: lint
lint: ## run tests with poetry (isort, black, pflake8, mypy)
	poetry run black .
	poetry run isort .
	poetry run pflake8 .
	# TODO: mypy 通す!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# poetry run mypy src --explicit-package-bases

.PHONY: setup-python
setup-python: ## setup python environment
	rm ./poetry.lock
	poetry env use python3.10
	poetry install --no-cache --no-interaction
	cp ./xfoil-python/libxfoil.dylib ./.venv/lib/python3.10/site-packages/xfoil/libxfoil.dylib