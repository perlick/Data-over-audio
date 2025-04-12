#ifndef DOA_MACROS
#define DOA_MACROS

#ifdef _WIN32
#define fcomplex _Fcomplex
#define sleep(n) Sleep((int)(n*1000))

#elif __linux
#define fcomplex float complex
#define _Cmulcr(x,y) x*y
#define _FCmulcr(x,y) x*y
#define _LCmulcr(x,y) x*y

#endif
#endif