CC=g++
CFLAGS=-std=c++11 -pthread

ROOT_DIR=$(PWD)

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${SRCS}
	${CC} ${CFLAGS} $< -o ${BINS}

run: all
	@$(ROOT_DIR)/${BINS} ${threadNum}

clean:
	@rm -f $(BINS)