# Get the directory where this Makefile lives
ROOT_DIR := $(CURDIR)

# Default target (so 'make' runs this)
.PHONY: all
all: gen_data

# Generate data
.PHONY: gen_data
gen_data:
	@echo "Generating data..."
	python3 $(ROOT_DIR)/data/generate_data/gen_maxcut_data.py --nbr_nodes $(nbr_nodes) --edge_mode $(edge_mode)
