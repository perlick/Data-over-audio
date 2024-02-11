all:
	. ../emsdk/emsdk_env.sh
	emcc -sWASM_WORKERS src/hello.c -o src/hello.js
