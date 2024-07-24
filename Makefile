OBJS=src/main.o src/circ_buf.o

all: $(OBJS)
	gcc -g $(OBJS) -lasound -lm -o src/main.out
test:
	gcc -g test/pcm_min.c -lasound -o test/pcm_min.out
	gcc -g test/sin-example.c -lasound -lm -o test/sin-example.out
