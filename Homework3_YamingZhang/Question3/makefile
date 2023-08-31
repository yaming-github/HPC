SHELL:=/bin/bash
CC=gcc
CFLAGS=-lopenblas

ROOT_DIR=$(PWD)

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${SRCS}
	@module load openblas/0.3.6
	@${CC} ${CFLAGS} $< -o ${BINS}

run: all
	@$(ROOT_DIR)/${BINS}

clean:
	@rm -f $(BINS)