#ifdef _WIN32
#include <memoryapi.h>

void* create_shared_memory(size_t size){
    int allocation = MEM_COMMIT | MEM_RESERVE;
    int protection = PAGE_READWRITE;
    return VirtualAlloc(NULL, size, allocation, protection);
}

#elif __linux
#include <sys/mman.h>
#include <stddef.h>

void* create_shared_memory(size_t size){
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_SHARED | MAP_ANONYMOUS;
    return mmap(NULL, size, protection, visibility, -1, -0);
}

#endif
