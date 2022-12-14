#include "field377.hpp"

const uint64_t Zero[6] = {
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};
const uint64_t One[6] = {
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};
const uint64_t P[6] = {
    0x8508c00000000001,
    0x170b5d4430000000,
    0x1ef3622fba094800,
    0x1a22d9f300f5138f,
    0xc63b05c06ca1493b,
    0x01ae3a4617c510ea
};
const uint64_t R[6] = {
    0x7af73fffffffffff,
    0xe8f4a2bbcfffffff,
    0xe10c9dd045f6b7ff,
    0xe5dd260cff0aec70,
    0x39c4fa3f935eb6c4,
    0x0051c5b9e83aef15
};
const uint64_t Ri[6] = {
    0x2b0909a28934f3a1,
    0x83264aa55c1cfac6,
    0x1accd49ca2a491ae,
    0xa28b2dce9e80e9a6,
    0x34d313ea126f7c08,
    0x0161de1ee3625456
};
const uint64_t R2[6] = {
    0x30832a73a1b25004,
    0xa404dcf0bcb14011,
    0xb7520b89a32a1bcc,
    0x6154e2a7cfb8d35a,
    0x475fc7349417d690,
    0x0155f398d8e0e30f
};

void random_mod(uint64_t res[6]){
    for(int i = 0; i < 5; i++) res[i] = ((uint64_t)rand()<<32) | rand();
    res[5] = (((uint64_t)rand()<<32) | rand()) & 0x01ae3a4617c510ea;
}

void add_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    add(res, a, b);
    if(geq(res, (uint64_t*)P)) {
        sub(res, res, (uint64_t*)P);
    }
}
void double_mod(uint64_t res[6], uint64_t a[6]) {
    lshift(res, a);
    if(geq(res, (uint64_t*)P)) {
        sub(res, res, (uint64_t*)P);
    }
}
void sub_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    if(geq(a, b)) {
        sub(res, a, b);
    } else {
        add(res, a, (uint64_t*)P);
        sub(res, res, b);
    }
}
void neg_mod(uint64_t res[6], uint64_t a[6]) {
    if(is_zero(a)) {
        copy(res, a);
    } else {
        sub(res, (uint64_t*)P, a);
    }
}
void mult_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
    memset(res, 0, 6*sizeof(uint64_t));
    for(uint32_t i=376; i!=UINT32_MAX; i--){
        uint32_t bit = i & 63;
        uint32_t word = i >> 6;
        double_mod(res, res);
        if(a[word] >> bit & 1){
            add_mod(res, res, b);
        }
    }
}
void inverse_mod(uint64_t res[6], uint64_t a[6]){
    uint64_t temp[6];
    inverse_mon(temp, a);
    mult_mon(res, temp, (uint64_t*)One);
}

void to_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)R2);
}
void from_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)One);
}

void mult_mon(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
    memset(res, 0, 6 * sizeof(uint64_t));
    uint64_t a0 = a[0] & 1;
    for(uint64_t i=0; i < 377; i++){
        uint64_t bit = i % 64;
        uint64_t word = i / 64;
        uint64_t qa = b[word] >> bit & 1;
        uint64_t qm = (res[0] & 1) ^ (a0 & qa);
        if(qa){
            add(res, res, a);
        }
        if(qm){
            add(res, res, (uint64_t*)P);
        }
        rshift(res, res);
    }
    if(geq(res, (uint64_t*)P)){
        sub(res, res, (uint64_t*)P);
    }
}
void inverse_mon(uint64_t res[6], uint64_t a[6]){
    memset(res, 0, 6 * sizeof(uint64_t));
    uint64_t u[6];
    uint64_t v[6];
    uint64_t s[6] = {1, 0, 0, 0, 0, 0};
    uint64_t r[6] = {0, 0, 0, 0, 0, 0};
    uint64_t k = 0;

    copy(u, (uint64_t*)P);
    copy(v, a);
    
    while(!is_zero(v)){
        if((u[0] & 1) == 0){
            rshift(u, u);
            lshift(s, s);
        }
        else if((v[0] & 1) == 0){
            rshift(v, v);
            lshift(r, r);
        }
        else if(!geq(v, u)){
            sub(u, u, v);
            rshift(u, u);
            add(r, r, s);
            lshift(s, s);
        }
        else{
            sub(v, v, u);
            rshift(v, v);
            add(s, s, r);
            lshift(r, r);
        }
        k++;
    }
    if(geq(r, (uint64_t*)P)){
        sub(r, r, (uint64_t*)P);
    }

    sub(r, (uint64_t*)P, r);
    k -= 377;

    for(int i = 0; i < k; i++){
        if(r[0] & 1 == 1){
            add(r, r, (uint64_t*)P);
        }
        rshift(r, r);
    }
    copy(res, r);
}

