#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdexcept>

void print(uint32_t a[12]) {
    for (int i = 11; i >=0; i--) {
        printf("%08X ", a[i]);
        if(i%4==0) printf("\n");
    }
    printf("\n");
}

///

bool is_negative(uint32_t a[12]) {
    return a[11] & 0x80000000;
}

bool is_zero(uint32_t a[12]) {
    for (int i = 0; i < 12; i++) {
        if (a[i] != 0) {
            return false;
        }
    }
    return true;
}

bool geq(uint32_t a[12], uint32_t b[12]) {
    bool sign_a = is_negative(a);
    bool sign_b = is_negative(b);
    if (sign_a != sign_b) {
        return !sign_a;
    }
    for (int i = 11; i >= 0; i--) {
        if (a[i] != b[i]) {
            return sign_a ? a[i] < b[i] : a[i] > b[i];
        }
    }
    return true;
}

bool eq(uint32_t a[12], uint32_t b[12]) {
    for (int i = 11; i >= 0; i--) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int lzeroes(uint32_t a[12]) {
    int zeroes = 0;
    for(int i = 11; i >= 0; i--) {
        if(a[i]==0) zeroes += 32;
        else {
            zeroes += __builtin_clzll(a[i]);
            break;
        }
    }
    return zeroes;
}

int degree(uint32_t a[12]) {
    return 384 - lzeroes(a); //12 *32
}

///

void negate(uint32_t res[12], uint32_t a[12]){
    for(int i = 0; i < 12; i++) {
        res[i] = ~a[i];
    }
    for(int i = 0; i < 12; i++) {
        if(res[i] == UINT64_MAX) {
            res[i] == 0;
        }
        else {
            res[i]++;
            break;
        }
    }
}

void lshift(uint32_t res[12], uint32_t a[12], int shift){
    if(shift == 0){
        memcpy(res, a, 12*sizeof(uint32_t));
        return;
    }
    int bit_shift = shift % 32;
    int word_shift = shift / 32;
    for (int i = 11; i >= 0; i--) {
        res[i] = (a[i] << shift);
        if (i > 0) res[i] |= (a[i-1] >> (32-shift));
    }
    for (int i = 11; i >= 0; i--){
        if(i - word_shift >= 0) res[i] = res[i - word_shift];
        else res[i] = 0;
    }
}

void rshift(uint32_t res[12], uint32_t a[12], int shift){
    int bit_shift = shift % 32;
    int word_shift = shift / 32;
    for (int i = 0; i < 12; i++) {
        res[i] = (a[i] >> shift);
        if (i < 11) res[i] |= (a[i+1] << (32-shift));
    }
    for (int i = 0; i < 12; i++){
        if(i + word_shift < 12) res[i] = res[i + word_shift];
        else res[i] = 0;
    }
}

bool add(uint32_t res[12], uint32_t a[12], uint32_t b[12]){
    uint64_t carry = 0;
    for (int i = 0; i < 12; i++) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        res[i] = sum & 0xFFFFFFFF;
        carry = sum >> 32;
    }
    return carry;
}

bool sub(uint32_t res[12], uint32_t a[12], uint32_t b[12]){
    uint64_t carry = 0;
    for (int i = 0; i < 12; i++) {
        uint64_t sum = (uint64_t)a[i] - (uint64_t)b[i] - carry;
        res[i] = sum & 0xFFFFFFFF;
        carry = sum >> 32;
    }
    return carry;
}


int main(){
    uint32_t a[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0x80000000}; // -11
    uint32_t b[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0x80000000}; // -1
    uint32_t c[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // 0
    
    add(c, a, b);
    print(c);
}
