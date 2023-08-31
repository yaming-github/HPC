CC=g++
CFLAGS=-pthread -fopenmp

ROOT_DIR=$(PWD)

pthread=Q2_pthread.c
omp=Q2_omp.c
pthreadB=pthread
ompB=omp
SRCS := $(wildcard *.c)

all: ${SRCS}
	@${CC} ${CFLAGS} ${pthread} -o ${pthreadB}
	@${CC} ${CFLAGS} ${omp} -o ${ompB}

run: all
	@printf "Running with pthread...\n"
	@$(ROOT_DIR)/${pthreadB} ${threadNum} ${dartNum}
	@printf "\nRunning with omp...\n"
	@$(ROOT_DIR)/${ompB} ${threadNum} ${dartNum}

clean:
	@rm -f $(BINS)