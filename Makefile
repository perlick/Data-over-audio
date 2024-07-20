all:
	gcc -g src/main.c -lasound  -o src/main.out
test:
	gcc -g test/pcm_min.c -lasound -o test/pcm_min.out
