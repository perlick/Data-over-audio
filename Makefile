all:
	gcc -g src/main.c -lasound -lm -o src/main.out
test:
	gcc -g test/pcm_min.c -lasound -o test/pcm_min.out
	gcc -g test/sin-example.c -lasound -lm -o test/sin-example.out
