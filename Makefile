# Format source code automatically
style:
	black src/mt_benchmarks tests
	isort src/mt_benchmarks tests

# Control quality
quality:
	black --check --diff src/mt_benchmarks tests
	isort --check-only src/mt_benchmarks tests
	flake8 src/mt_benchmarks tests

# Run tests
test:
	pytest
