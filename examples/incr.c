#include <stdint.h>
#include <stdio.h>
extern int64_t init();
extern int64_t incrtwice();

int main(void) {
    init();
    printf("incr 2 result: %d", incrtwice());
}