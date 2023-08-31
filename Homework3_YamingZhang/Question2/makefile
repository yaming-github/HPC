CC=gcc
CFLAGS=-fopenmp

ROOT_DIR=$(PWD)

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${SRCS}
	@${CC} ${CFLAGS} $< -o ${BINS}

run: all
	@$(ROOT_DIR)/${BINS}

clean:
	@rm -f $(BINS)