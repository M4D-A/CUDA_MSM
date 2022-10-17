#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdexcept>

uint64_t P[6] = {
    0x8508c00000000001,
    0x170b5d4430000000,
    0x1ef3622fba094800,
    0x1a22d9f300f5138f,
    0xc63b05c06ca1493b,
    0x01ae3a4617c510ea
};

uint64_t R[6] = {
    0x7af73fffffffffff,
    0xe8f4a2bbcfffffff,
    0xe10c9dd045f6b7ff,
    0xe5dd260cff0aec70,
    0x39c4fa3f935eb6c4,
    0x0051c5b9e83aef15
};

uint64_t Ri[6] = {
    0x2b0909a28934f3a1,
    0x83264aa55c1cfac6,
    0x1accd49ca2a491ae,
    0xa28b2dce9e80e9a6,
    0x34d313ea126f7c08,
    0x0161de1ee3625456
};

uint64_t R2[6] = {
    0x30832a73a1b25004,
    0xa404dcf0bcb14011,
    0xb7520b89a32a1bcc,
    0x6154e2a7cfb8d35a,
    0x475fc7349417d690,
    0x0155f398d8e0e30f
};

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

void print(uint64_t a[6]) {
    printf("0x");
    for (int i = 5; i >=0; i--) {
        printf("%016lX", a[i]);
        if(i != 0) printf("_");
    }
    printf("\n");
}

bool geq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
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

bool is_zero(uint64_t a[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != 0) {
            return false;
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
        }
    }
    return result;
}

void copy(uint64_t res[6], uint64_t a[6]){
    for(int i=0; i<6; i++) res[i] = a[i];
}

void rshift(uint64_t res[6], uint64_t a[6], int shift){
    int bit_shift = shift % 64;
    int word_shift = shift / 64;
    for (int i = 0; i < 6; i++) {
        res[i] = (a[i] >> shift);
        if (i < 5) res[i] |= (a[i+1] << (64-shift));
    }
    for (int i = 0; i < 6; i++){
        if(i + word_shift < 6) res[i] = res[i + word_shift];
        else res[i] = 0;
    }
}

void lshift(uint64_t res[6], uint64_t a[6], int shift){
    int bit_shift = shift % 64;
    int word_shift = shift / 64;
    for (int i = 5; i >= 0; i--) {
        res[i] = (a[i] << shift);
        if (i > 0) res[i] |= (a[i-1] >> (64-shift));
    }
    for (int i = 5; i >= 0; i--){
        if(i - word_shift >= 0) res[i] = res[i - word_shift];
        else res[i] = 0;
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

void addMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    add(res, a, b);
    if(geq(res, P)) {
        sub(res, res, P);
    }
}

void subMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    if(geq(a, b)) {
        sub(res, a, b);
    } else {
        add(res, a, P);
        sub(res, res, b);
    }
}

void negMod(uint64_t res[6], uint64_t a[6]) {
    if(is_zero(a)) {
        copy(res, a);
    } else {
        sub(res, P, a);
    }
}

void prodMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
    memset(res, 0, 6*sizeof(uint64_t));
    for(int i=376; i>=0; i--){
        int bit = i % 64;
        int word = i / 64;
        addMod(res, res, res);
        if(a[word] >> bit & 1){
            addMod(res, res, b);
        }
    }
}

void prodMon(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
    memset(res, 0, 6*sizeof(uint64_t));
    int a0 = a[0] & 1;
    for(int i=0; i < 377; i++){
        int bit = i % 64;
        int word = i / 64;
        int qa = b[word] >> bit & 1;
        int qm = (res[0] & 1) ^ (a0 & qa);
        if(qa){
            add(res, res, a);
        }
        if(qm){
            add(res, res, P);
        }
        rshift(res, res, 1);
    }
    if(geq(res, P)){
        sub(res, res, P);
    }
}

void inverseMon(uint64_t res[6], uint64_t a[6]){
    uint64_t u[6];
    uint64_t v[6];
    uint64_t s[6] = {1,0,0,0,0,0};
    uint64_t r[6] = {0,0,0,0,0,0};
    uint64_t k = 0;

    copy(u, P);
    copy(v, a);
    
    while(!is_zero(v)){
        if((u[0] & 1) == 0){
            rshift(u, u, 1);
            lshift(s, s, 1);
        
        }
        else if((v[0] & 1) == 0){
            rshift(v, v, 1);
            lshift(r, r, 1);
        }
        else if(!geq(v, u)){
            sub(u, u, v);
            rshift(u, u, 1);
            add(r, r, s);
            lshift(s, s, 1);
        }
        else{
            sub(v, v, u);
            rshift(v, v, 1);
            add(s, s, r);
            lshift(r, r, 1);
        }
        k++;
    }
    if(geq(r, P)){
        sub(r, r, P);
    }

    sub(r, P, r);
    k -= 377;

    for(int i = 0; i < k; i++){
        if(r[0] & 1 == 1){
            add(r, r, P);
        }
        rshift(r, r, 1);
    }
    copy(res, r);
}

void inverseMod(uint64_t res[6], uint64_t a[6]){
    uint64_t r[6] = {0,0,0,0,0,0};
    inverseMon(r, a);
    prodMon(res, r, One);
}

bool isPoint(uint64_t p[12]){
    uint64_t y2[6];
    prodMod(y2, p+6, p+6);

    uint64_t x2[6], x3[6];
    prodMod(x2, p, p);
    prodMod(x3, x2, p);
    addMod(x3,x3,One);
    return eq(x3, y2);
}

void copyP(uint64_t res[12], uint64_t p[12]){
    for(int i = 0; i < 12; i++) res[i] = p[i];
}

void doubleP(uint64_t res[12], uint64_t p[12]){
    uint64_t s[6], t[6];

    addMod(res, p+6, p+6); // 2y
    inverseMod(res+6, res);  // 1/2y <- 

    prodMod(res, p, p); // x^2
    addMod(s, res, res); // 2x^2
    addMod(res, res, s); // 3x^2
    
    prodMod(s, res, res+6); // 3x^2/2y
    prodMod(res, s, s); // s^2

    subMod(res, res, p); // s^2 - x
    subMod(res, res, p); // s^2 - 2x = xr

    subMod(t, res, p); // xr - x
    prodMod(res+6, s, t); // s(xr - x)
    addMod(res+6, res+6, p+6); // s(xr - x) - y
    negMod(res+6, res+6); // y' = -(s(xr - x) - y)
}

void addP(uint64_t res[12], uint64_t p[12], uint64_t q[12]){
    uint64_t s[6], t[6];

    if(is_zero(p) && is_zero(p+6)){
        copyP(res, q);
        return;
    }

    else if(is_zero(q) && is_zero(q+6)){
        copyP(res, p);
        return;
    }

    else if(eq(p, q)){
        if(eq(p+6, q+6)){
            doubleP(res, p);
        } else {
            printf("p = -q\n");
            memset(res, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    subMod(res, q, p); // xq - xp
    inverseMod(res+6, res); // 1/(xq - xp)

    subMod(res, q+6, p+6); // yq - yp

    prodMod(s, res, res+6); // s = (yq - yp)/(xq - xp)

    prodMod(res, s, s); // s^2
    subMod(res, res, p); // s^2 - xp
    subMod(res, res, q); // s^2 - xp - xq = xr

    subMod(t, res, p); // xr - xp
    prodMod(res+6, s, t); // s(xr - xp)
    addMod(res+6, res+6, p+6); // s(xr - xp) + yp
    negMod(res+6, res+6); // y' = -(s(xr - xp) + yp)
}

void negP(uint64_t res[12], uint64_t p[12]){
    copy(res, p);
    negMod(res+6, p+6);
}

void scalarP(uint64_t res[12], uint64_t p[12], uint64_t k[6]){
    uint64_t q[12], tq[12], tres[12];
    memset(res, 0, 12*sizeof(uint64_t));
    copyP(q, p);
    uint64_t bits = 384 - leading_zeros(k);
    printf("bits = %lu\n", bits);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            addP(tres, res, q);
            copy(res, tres);
            copy(res + 6, tres + 6);
        }

        doubleP(tq, q);
        copyP(q, tq);
    }
}



int main(){
    uint64_t p[12] = {
        0xeab9b16eb21be9ef,
        0xd5481512ffcd394e,
        0x188282c8bd37cb5c,
        0x85951e2caa9d41bb,
        0xc8fc6225bf87ff54,
        0x008848defe740a67,

        0xfd82de55559c8ea6,
        0xc2fe3d3634a9591a,
        0x6d182ad44fb82305,
        0xbd7fb348ca3e52d9,
        0x1f674f5d30afeec4,
        0x01914a69c5102eff,
    };

    uint64_t k[6] = {
        0x0000000000041111,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    };

    uint64_t res[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    uint64_t temp[12];
    uint64_t res2[12] = {0,0,0,0,0,0,0,0,0,0,0,0};


    for(int i = 0; i < 0x41111; i++){
        addP(temp, p, res);
        copyP(res, temp);
    }

    scalarP(res2, p, k);
    print(res);
    print(res+6);
    print(res2);
    print(res2+6);
}