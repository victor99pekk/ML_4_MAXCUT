# Get the directory where this Makefile lives
ROOT_DIR := $(CURDIR)
datatype ?= debug

# Default target (so 'make' runs this)
.PHONY: all
all: gen_data

# Generate data
.PHONY: gen_data
gen_data:
	python3 $(ROOT_DIR)/data/gen_maxcut_data.py --nbr_nodes $(nbr_nodes) --edge_mode $(edge_mode) --datatype $(datatype)
