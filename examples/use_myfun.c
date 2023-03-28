#include <stdint.h>
#include <stdio.h>
extern int64_t fib(int64_t);

int main(void) {
    printf("myfun result: %d", fib(6));
}