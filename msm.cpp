#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

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

void lshift(uint64_t *result, uint64_t* a, int shift, int n){
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

void div(uint64_t* quotient, uint64_t* residue, uint64_t* a, uint64_t* b, int n) {
    memset(quotient, 0, n * sizeof(uint64_t));
    memcpy(residue, a, n * sizeof(uint64_t));

    uint64_t ta = 0;
    for(int i = n-1; i >= 0; i--) {
        if(b[i]==0) ta+=64;
        else {
            ta += __builtin_clzll(b[i]);
            break;
        }
    }
    uint64_t tb = 0;
    for(int i = n-1; i >= 0; i--) {
        if(a[i]==0) tb+=64;
        else {
            tb += __builtin_clzll(a[i]);
            break;
        }
    }
    

    int64_t dt = ta - tb;
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

int main(){
    uint64_t A[3] = {
        0xdeadbeeffeedface,
        0xfacefeedb3332222, 
        0xc00feebadbadf00d
    };

    uint64_t B[3] = {
        0xdeadbeeffeedface, 
        0x0acefeedb3332222, 
        0x0000000000000000
    };

    uint64_t R[3] = {0,0,0};
    uint64_t Q[3] = {0,0,0};


    for(int i = 0; i < 20; i++){
        div(Q, R, A, B, 3);
        for(int j = 0; j < 3; j++){
            printf("%016lX ", R[2-j]);
        }
        printf("\n");
        for(int j = 0; j < 3; j++){
            printf("%016lX ", Q[2-j]);
        }        
        printf("\n\n");
        memcpy(A, B, 3 * sizeof(uint64_t));
        memcpy(B, R, 3 * sizeof(uint64_t));
    }


    printf("\n");
}
