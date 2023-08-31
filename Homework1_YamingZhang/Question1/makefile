CC=gcc
PP=g++
OFLAGS=-O3
CFLAGS=-std=c++11

ROOT_DIR=$(PWD)

LINPACK=Q1_linpack.c
INTEGER=Q1_integer_sum.c
LIST=Q1_list_traversal.c
LINPACKBIN=Q1_linpack
INTEGERBIN=Q1_integer_sum
LISTBIN=Q1_list_traversal
LINPACKBINO=Q1_linpackO
INTEGERBINO=Q1_integer_sumO
LISTBINO=Q1_list_traversalO
SRCS := $(wildcard *.c)

all: ${SRCS}
	${CC} ${LINPACK} -o ${LINPACKBIN}
	${CC} ${OFLAGS} ${LINPACK} -o ${LINPACKBINO}
	${PP} ${CFLAGS} ${INTEGER} -o ${INTEGERBIN}
	${PP} ${CFLAGS} ${OFLAGS} ${INTEGER} -o ${INTEGERBINO}
	${PP} ${CFLAGS} ${LIST} -o ${LISTBIN}
	${PP} ${CFLAGS} ${OFLAGS} ${LIST} -o ${LISTBINO}

run: all
	@$(ROOT_DIR)/${LINPACKBIN} || /bin/true
	@printf "\n"
	@$(ROOT_DIR)/${INTEGERBIN} || /bin/true
	@printf "\n"
	@$(ROOT_DIR)/${LISTBIN}

optimize: all
	@$(ROOT_DIR)/${LINPACKBINO} || /bin/true
	@printf "\n"
	@$(ROOT_DIR)/${INTEGERBINO} || /bin/true
	@printf "\n"
	@$(ROOT_DIR)/${LISTBINO}

clean:
	@rm -f ${LINPACKBIN} ${LINPACKBINO} ${INTEGERBIN} ${INTEGERBINO} ${LISTBIN} ${LISTBINO}