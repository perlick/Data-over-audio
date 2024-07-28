OBJS=src/main.o src/circ_buf.o src/filter.o

all: $(OBJS)
	gcc -g $(OBJS) -lasound -lfftw3 -lm -o src/main.out
test:
	gcc -g test/pcm_min.c -lasound -o test/pcm_min.out
	gcc -g test/sin-example.c -lasound -lm -o test/sin-example.out
	gcc -g test/cap_min.c -lasound -o test/cap_min.out
