OBJS=src/main.c src/circ_buf.c src/filter.c src/linux_asoundlib.c src/proc.c src/shared_mem.c

all: $(OBJS)
	gcc -g $(OBJS) -lasound -lfftw3 -lm -o src/main.out
test:
	gcc -g test/pcm_min.c -lasound -o test/pcm_min.out
	gcc -g test/sin-example.c -lasound -lm -o test/sin-example.out
	gcc -g test/cap_min.c -lasound -o test/cap_min.out
