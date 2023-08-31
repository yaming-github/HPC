CC=g++
CFLAGS=-pthread

ROOT_DIR=$(PWD)

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${SRCS}
	@${CC} ${CFLAGS} $< -o ${BINS}

run: all
	@$(ROOT_DIR)/${BINS} ${threadNum} ${dartNum}

clean:
	@rm -f $(BINS)