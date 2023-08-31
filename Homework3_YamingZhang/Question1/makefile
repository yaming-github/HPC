SHELL:=/bin/bash
CC=g++
CFLAGS=-std=c++11 -march=native

ROOT_DIR=$(PWD)

Q1=Q1.cpp
Q1BIN=Q1
Q12=Q1_2.cpp
Q12BIN=Q1_2

all: ${SRCS}
	@module add gcc/10.1.0
	@${CC} ${CFLAGS} ${Q1} $< -o ${Q1BIN}
	@${CC} ${CFLAGS} ${Q12} $< -o ${Q12BIN}

run: all
	@$(ROOT_DIR)/${Q1BIN}
	@printf "\n"
	@$(ROOT_DIR)/${Q12BIN}

clean:
	@rm -f $(Q1BIN) $(Q12BIN)