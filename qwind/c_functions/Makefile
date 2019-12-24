CC=gcc
OBJS = qwind_library.so
CFLAGS =  -Wall -fPIC -g -shared
INCLUDE = *.c
LIBS = -lm -lgsl -lgslcblas

make:
	${CC} ${CFLAGS} ${INCLUDE} -o ${OBJS} ${LIBS}

clean:
	-rm -f *.so
