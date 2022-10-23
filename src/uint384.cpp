#include "uint384.hpp"

uint64_t Zero[6] = {
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};

uint64_t One[6] = {
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};

uint64_t random64(){
    uint64_t res = rand();
    res = res << 32;
    res = res | rand();
    return res;
}

void random384(uint64_t res[6]){
    for(int i = 0; i < 6; i++) res[i] = random64();
}

void print(uint64_t a[6]) {
    printf("0x");
    for (int i = 5; i >=0; i--) {
        printf("%016lX", a[i]);
        if(i != 0) printf("_");
    }
    printf("\n");
}

bool is_zero(uint64_t a[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != 0) {
            return false;
        }
    }
    return true;
}

bool eq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool geq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return true;
}

uint64_t leading_zeros(uint64_t a[6]) {
    uint64_t result = 0;
    for (int i = 5; i >= 0; i--) {
        if (a[i] == 0) {
            result += 64;
        } else {
            result += __builtin_clzll(a[i]);
            break;
        }
    }
    return result;
}

void copy(uint64_t res[6], uint64_t a[6]){
    memcpy(res, a, 6 * sizeof(uint64_t));
}

void rshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 0; i < 6; i++) {
        res[i] = (a[i] >> 1);
        if (i < 5) res[i] |= (a[i+1] << (63));
    }
}

void lshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 5; i >= 0; i--) {
        res[i] = (a[i] << 1);
        if (i > 0) res[i] |= (a[i-1] >> (63));
    }
}

bool add(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] + b[i] + carry;
        res[i] = sum;
        carry = sum < ai;
    }
    return carry;
}

bool sub(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] - b[i] - carry;
        res[i] = sum;
        carry = sum > ai;
    }
    return carry;
}