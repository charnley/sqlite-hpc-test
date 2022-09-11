
all: env

# Python

env:
	mamba env create -f ./environment.yml -p ./env --quiet

run:
	python -m program.submit --n-entries 10_000 --n-jobs 50
