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

bool eq(uint64_t a[6], uint32_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
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
        printf("aaa\n");
        sub(res, a, b);
    } else {
        printf("bbb\n");
        add(res, a, P);
        sub(res, res, b);
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

int main(){
    uint64_t a[6] = {
        0x6363e703c89167cc,
        0x83368d7243d7e492,
        0x42ee81675b84219b,
        0x49a9193f7d21d636,
        0xaf5a0dfc4f53d7bc,
        0x128180471539142,
    };
    uint64_t b[6] = {
        0x5bf1af2de2eb371d,
        0x5f4083ef8f614993,
        0x2eb35f8a0cb06c48,
        0x93469109d71ea0e7,
        0x71b544c329b5a0bf,
        0x11db9d324ccc864,
    };

    uint64_t res[6];

    for(int i = 0; i < 1000000; i++){
        prodMon(res, a, b);
        copy(b,a);
        copy(a,res);
    }
    print(a);
    print(b);
}