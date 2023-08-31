CC=g++
CFLAGS=-pthread

ROOT_DIR=$(PWD)

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${SRCS}
	@${CC} ${CFLAGS} $< -o ${BINS}

run: all
	@$(ROOT_DIR)/${BINS} ${pNum}

clean:
	@rm -f $(BINS)