#include "bls12-377.hpp"

uint64_t G1[12] = {
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


void printP(uint64_t P[12]){
    print(P);
    print(P+6);
}

bool isPoint(uint64_t P[12]){
    uint64_t y2[6];
    prodMod(y2, P+6, P+6);

    uint64_t x2[6], x3[6];
    prodMod(x2, P, P);
    prodMod(x3, x2, P);
    addMod(x3,x3,One);
    return eq(x3, y2);
}

bool isZeroP(uint64_t P[12]){
    return is_zero(P) && is_zero(P+6);
}

void randomP(uint64_t res[12]){
    uint64_t k[6];
    randomMod(k);
    scalarP(res, G1, k);
}

void copyP(uint64_t R[12], uint64_t P[12]){
    for(int i = 0; i < 12; i++) R[i] = P[i];
}

void doubleP(uint64_t R[12], uint64_t P[12]){
    uint64_t s[6], t[6];

    addMod(R, P+6, P+6); // 2y
    inverseMod(R+6, R);  // 1/2y <- 

    prodMod(R, P, P); // x^2
    addMod(s, R, R); // 2x^2
    addMod(R, R, s); // 3x^2
    
    prodMod(s, R, R+6); // 3x^2/2y
    prodMod(R, s, s); // s^2

    subMod(R, R, P); // s^2 - x
    subMod(R, R, P); // s^2 - 2x = xr

    subMod(t, R, P); // xr - x
    prodMod(R+6, s, t); // s(xr - x)
    addMod(R+6, R+6, P+6); // s(xr - x) - y
    negMod(R+6, R+6); // y' = -(s(xr - x) - y)
}

void addP(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
    uint64_t s[6], t[6];

    if(is_zero(P) && is_zero(P+6)){
        copyP(R, Q);
        return;
    }

    else if(is_zero(Q) && is_zero(Q+6)){
        copyP(R, P);
        return;
    }

    else if(eq(P, Q)){
        if(eq(P+6, Q+6)){
            doubleP(R, P);
        } else {
            printf("p = -q\n");
            memset(R, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    subMod(R, Q, P); // xq - xp
    inverseMod(R+6, R); // 1/(xq - xp)

    subMod(R, Q+6, P+6); // yq - yp

    prodMod(s, R, R+6); // s = (yq - yp)/(xq - xp)

    prodMod(R, s, s); // s^2
    subMod(R, R, P); // s^2 - xp
    subMod(R, R, Q); // s^2 - xp - xq = xr

    subMod(t, R, P); // xr - xp
    prodMod(R+6, s, t); // s(xr - xp)
    addMod(R+6, R+6, P+6); // s(xr - xp) + yp
    negMod(R+6, R+6); // y' = -(s(xr - xp) + yp)
}

void negP(uint64_t R[12], uint64_t P[12]){
    copy(R, P);
    negMod(R+6, P+6);
}

void scalarP(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
    if(isZeroP(P) || is_zero(k)){
        memset(R, 0, 12*sizeof(uint64_t));
        return;
    }

    uint64_t Q[12], tQ[12], tR[12];
    memset(R, 0, 12*sizeof(uint64_t));
    copyP(Q, P);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            addP(tR, R, Q);
            copyP(R, tR);
        }

        doubleP(tQ, Q);
        copyP(Q, tQ);
    }
}

void msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(R, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        memset(T1, 0, 12*sizeof(uint64_t));
        scalarP(T1, P + 12*i, k + 6*i);
        addP(T2, R, T1);
        copyP(R, T2);
    }
}

void toMonP(uint64_t R[12], uint64_t P[12]){
    toMon(R, P);
    toMon(R+6, P+6);
}

void fromMonP(uint64_t R[12], uint64_t P[12]){
    fromMon(R, P);
    fromMon(R+6, P+6);
}

void doublePMon(uint64_t R[12], uint64_t P[12]){
    uint64_t s[6], t[6];

    addMod(R, P+6, P+6); // 2y
    inverseMon(t, R);  // 1/2y <- s
    toMon(R+6, t);

    prodMon(R, P, P); // x^2
    addMod(s, R, R); // 2x^2
    addMod(R, R, s); // 3x^2
    
    prodMon(s, R, R+6); // 3x^2/2y
    prodMon(R, s, s); // s^2

    subMod(R, R, P); // s^2 - x
    subMod(R, R, P); // s^2 - 2x = xr

    subMod(t, R, P); // xr - x
    prodMon(R+6, s, t); // s(xr - x)
    addMod(R+6, R+6, P+6); // s(xr - x) - y
    negMod(R+6, R+6); // y' = -(s(xr - x) - y)
}

void addPMon(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
    uint64_t s[6], t[6];

    if(is_zero(P) && is_zero(P+6)){
        copyP(R, Q);
        return;
    }

    else if(is_zero(Q) && is_zero(Q+6)){
        copyP(R, P);
        return;
    }

    else if(eq(P, Q)){
        if(eq(P+6, Q+6)){
            doublePMon(R, P);
        } else {
            printf("p = -q\n");
            memset(R, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    subMod(R, Q, P); // xq - xp
    inverseMon(t, R);  // 1/2y <- s
    toMon(R+6, t);

    subMod(R, Q+6, P+6); // yq - yp

    prodMon(s, R, R+6); // s = (yq - yp)/(xq - xp)

    prodMon(R, s, s); // s^2
    subMod(R, R, P); // s^2 - xp
    subMod(R, R, Q); // s^2 - xp - xq = xr

    subMod(t, R, P); // xr - xp
    prodMon(R+6, s, t); // s(xr - xp)
    addMod(R+6, R+6, P+6); // s(xr - xp) + yp
    negMod(R+6, R+6); // y' = -(s(xr - xp) + yp)
}

void scalarPMon(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
    uint64_t Q[12], tQ[12], tR[12];
    memset(R, 0, 12*sizeof(uint64_t));
    copyP(Q, P);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            addPMon(tR, R, Q);
            copyP(R, tR);
        }
        doublePMon(tQ, Q);
        copyP(Q, tQ);
    }
}

void msmMon(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(R, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        memset(T1, 0, 12*sizeof(uint64_t));
        scalarPMon(T1, P + 12*i, k + 6*i);
        addPMon(T2, R, T1);
        copyP(R, T2);
    }
}

void msm2(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(R, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        toMonP(T1, P + 12*i);
        scalarPMon(T2, T1, k + 6*i);
        addPMon(T1, R, T2);
        copyP(R, T1);
    }
    fromMonP(T1, R);
    copyP(R, T1);
}

void bucketMSM(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    memset(R, 0, 12*sizeof(uint64_t));
    
    
    
    uint64_t buckets[65536][12];
    uint64_t temp_sum[12], temp[12], bucket_sum[12];


    for(uint64_t w = 0; w < 6; w++){
        for(uint64_t b = 0; b < 4; b++){
            printf("w = %lu, b = %lu\n", w, b);
            memset(buckets, 0, 65536*12*sizeof(uint64_t));
            memset(temp_sum, 0, 12*sizeof(uint64_t));
            memset(temp, 0, 12*sizeof(uint64_t));
            memset(bucket_sum, 0, 12*sizeof(uint64_t));

            uint64_t shift = b*16;
            for(uint64_t i = 0; i < n; i++){
                uint64_t index = k[6*i + w] >> shift & 0xFFFF;
                addP(temp, buckets[index], P + 12*i);
                copyP(buckets[index], temp);
            }
            
            memset(temp_sum, 0, 12*sizeof(uint64_t));
            memset(temp, 0, 12*sizeof(uint64_t));

            for(uint64_t i = 65535; i >= 1; i--){
                addP(temp, temp_sum, buckets[i]);
                copyP(temp_sum, temp);

                addP(temp, bucket_sum, temp_sum);
                copyP(bucket_sum, temp);
            }
            uint64_t shift_mul[6] = {0, 0, 0, 0, 0, 0,};
            shift_mul[w] = 1lu << shift;
            scalarP(temp, bucket_sum, shift_mul);
            addP(temp_sum, R, temp);
            copyP(R, temp_sum);
        }
    }
}