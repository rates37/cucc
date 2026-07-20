
PYTHON ?= python3
.PHONY: test test-unit test-integration test_heavy bench

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-unit:
	$(PYTHON) -m unittest discover -v tests.test_transpiler

test-integration:
	$(PYTHON) -m unittest -v tests.test_toolchain

test-heavy:
	CUCC_RUN_HEAVY=1 $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

bench:
	./bin/cucc bench/pool_bench.cu -O2 -o /tmp/cucc_pool_bench
	/tmp/cucc_pool_bench
