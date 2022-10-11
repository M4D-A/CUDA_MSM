#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdexcept>



bool geq(uint64_t *a, uint64_t *b, int n) {
    for (int i = n-1; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

bool leq(uint64_t *a, uint64_t *b, int n) {
    for (int i = n-1; i >= 0; i--) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return true;
}

bool eq(uint64_t *a, uint64_t *b, int n) {
    for (int i = n-1; i >= 0; i--) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

bool is_zero(uint64_t *a, int n) {
    for (int i = n-1; i >= 0; i--) {
        if (a[i] != 0) return false;
    }
    return true;
}

bool is_one(uint64_t *a, int n) {
    for (int i = n-1; i >= 1; i--) {
        if (a[i] != 0) return false;
    }
    return a[0] == 1;
}

int lzeroes(uint64_t *a, int n) {
    uint64_t zeroes = 0;
    for(int i = n-1; i >= 0; i--) {
        if(a[i]==0) zeroes += 64;
        else {
            zeroes += __builtin_clzll(a[i]);
            break;
        }
    }
    return zeroes;
}

int degree(uint64_t *a, int n) {
    return n*64 - lzeroes(a, n);
}

void negate(uint64_t *result, uint64_t *a, int n){
    for(int i = 0; i < n; i++) {
        result[i] = ~a[i];
    }
    for(int i = 0; i < n; i++) {
        if(result[i] == UINT64_MAX) {
            result[i] == 0;
        }
        else {
            result[i]++;
            break;
        }
    }
}

void lshift(uint64_t *result, uint64_t* a, int shift, int n){
    if(shift == 0){
        memcpy(result, a, n*sizeof(uint64_t));
        return;
    }
    int bit_shift = shift % 64;
    int word_shift = shift / 64;
    for (int i = n-1; i >= 0; i--) {
        result[i] = (a[i] << shift);
        if (i > 0) result[i] |= (a[i-1] >> (64-shift));
    }
    for (int i = n-1; i >= 0; i--){
        if(i - word_shift >= 0) result[i] = result[i - word_shift];
        else result[i] = 0;
    }
}

void rshift(uint64_t *result, uint64_t* a, int shift, int n){
    int bit_shift = shift % 64;
    int word_shift = shift / 64;
    for (int i = 0; i < n; i++) {
        result[i] = (a[i] >> shift);
        if (i < n-1) result[i] |= (a[i+1] << (64-shift));
    }
    for (int i = 0; i < n; i++){
        if(i + word_shift < n) result[i] = result[i + word_shift];
        else result[i] = 0;
    }
}

bool add(uint64_t *result, uint64_t *a, uint64_t *b, int n) {
    uint64_t carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] + b[i] + carry;
        result[i] = sum;
        carry = sum < ai;
    }
    return carry;
}

bool sub(uint64_t *result, uint64_t *a, uint64_t *b, int n) {
    uint64_t carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t ai = a[i];
        uint64_t diff = a[i] - b[i] - carry;
        result[i] = diff;
        carry = diff > ai;
    }
    return carry;
}

void mul_sub(uint64_t *result, uint64_t *a, uint64_t *b, uint64_t *c, int n) {
    memcpy(result, a, (n)*sizeof(uint64_t));
    result[n] = 0;
    int m = degree(b, n);
    lshift(c, c, m, n);
    for (int i = m; i > 0; i--) {
        if (b[i/64] & (1ULL << (i%64))) {
            int carry = sub(result, result, c, n);
            result[n] -= carry;
        }
        if(i>=0) rshift(c, c, 1, n);
    }
}

void div(uint64_t* quotient, uint64_t* residue, uint64_t* a, uint64_t* b, int n) {
    memset(quotient, 0, n * sizeof(uint64_t));
    memcpy(residue, a, n * sizeof(uint64_t));

    if (is_zero(b, n)) throw std::runtime_error("Division by zero");
    if(geq(b, a, n)) return;

    if (is_one(b, n)) {
        memcpy(quotient, a, n * sizeof(uint64_t));
        memset(residue, 0, n * sizeof(uint64_t));
        return;
    }

    if(eq(b, residue, n)) {
        quotient[0] = 1;
        memset(residue, 0, n * sizeof(uint64_t));
        return;
    }

    uint64_t ta = lzeroes(a, n);
    uint64_t tb = lzeroes(b, n);
    
    int64_t dt = tb - ta;
    lshift(b, b, dt, n);
    for(; dt >= 0; dt--) {
        if(geq(residue, b, n)) {
            sub(residue, residue, b, n);
            uint64_t q_word = dt / 64;
            uint64_t q_bit = dt % 64;
            quotient[q_word] |= (1ULL << q_bit);
        }
        if(dt > 0)rshift(b, b, 1, n);
    }
}

void egcd(uint64_t *x, uint64_t *y, uint64_t *d, uint64_t* a, uint64_t* b, int n) {
    uint64_t OR[8] = {0,0,0,0,0,0,0,0};
    uint64_t  R[8] = {0,0,0,0,0,0,0,0};
    memcpy(OR, a, n * sizeof(uint64_t));
    memcpy( R, b, n * sizeof(uint64_t));


    uint64_t OS[8] = {1,0,0,0,0,0,0,0};
    uint64_t  S[8] = {0,0,0,0,0,0,0,0};
    uint64_t OT[8] = {0,0,0,0,0,0,0,0};
    uint64_t  T[8] = {1,0,0,0,0,0,0,0};

    uint64_t Temp[8] = {0,0,0,0,0,0,0,0};
    uint64_t Q[8] = {0,0,0,0,0,0,0,0};

    while(!is_zero(R,n)){
        div(Q, Temp, OR, R, n);
        memcpy(OR, R, n * sizeof(uint64_t));
        memcpy(R, Temp, n * sizeof(uint64_t));

        mul_sub(Temp, OS, Q, S, n);
        memcpy(OS, S, n * sizeof(uint64_t));
        memcpy(S, Temp, n * sizeof(uint64_t));

        mul_sub(Temp, OT, Q, T, n);
        memcpy(OT, T, n * sizeof(uint64_t));
        memcpy(T, Temp, n * sizeof(uint64_t));

        print( Q, 4);
        print(OR, 4);
        print(OS, 4);
        print(OT, 4);
        print( R, 4);
        print( S, 4);
        print( T, 4);
        printf("\n");
    }

    memcpy(x, OS, (n+1) * sizeof(uint64_t));
    memcpy(y, OT, (n+1) * sizeof(uint64_t));
    memcpy(d, OR, (n+1) * sizeof(uint64_t));
}

void mod_add(uint64_t* result, uint64_t* a, uint64_t* b, uint64_t* mod, int n) {
    int64_t carry = add(result, a, b, n);
    int64_t ge = geq(result, mod, n);
    if (carry | ge) sub(result, result, mod, n);
}

void mod_double(uint64_t* result, uint64_t* a, uint64_t* mod, int n) {
    int64_t carry = a[n-1] >> 63;
    lshift(result, a, 1, n);
    int64_t ge = geq(result, mod, n);
    if (carry | ge) sub(result, result, mod, n);
}

void mod_mult(uint64_t* result, uint64_t* a, uint64_t* b, uint64_t* mod, int n) {
    memset(result, 0, n * sizeof(uint64_t));
    for (int i = n-1; i >=0; i--) {
        for (int j = 63; j >=0; j--) {
            mod_double(result, result, mod, n);
            if (a[i] & (1ULL << j)) {
                mod_add(result, result, b, mod, n);
            }
        }
    }
}




