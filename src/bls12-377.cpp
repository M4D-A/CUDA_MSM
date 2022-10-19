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


void print_P(uint64_t P[12]){
    print(P);
    print(P+6);
}

bool is_zero_P(uint64_t P[12]){
    return is_zero(P) && is_zero(P+6);
}

bool is_P(uint64_t P[12]){
    if(is_zero_P(P)) return true;
    uint64_t y2[6];
    mult_mod(y2, P+6, P+6);
    uint64_t x2[6], x3[6];
    mult_mod(x2, P, P);
    mult_mod(x3, x2, P);
    add_mod(x3,x3,One);
    return eq(x3, y2);
}

void random_P(uint64_t res[12]){
    uint64_t k[6];
    random_mod(k);
    scalar_P(res, G1, k);
}

void copy_P(uint64_t R[12], uint64_t P[12]){
    for(int i = 0; i < 12; i++) R[i] = P[i];
}

void double_P(uint64_t R[12], uint64_t P[12]){
    uint64_t s[6], t[6];

    add_mod(R, P+6, P+6); // 2y
    inverse_mod(R+6, R);  // 1/2y <- 

    mult_mod(R, P, P); // x^2
    add_mod(s, R, R); // 2x^2
    add_mod(R, R, s); // 3x^2
    
    mult_mod(s, R, R+6); // 3x^2/2y
    mult_mod(R, s, s); // s^2

    sub_mod(R, R, P); // s^2 - x
    sub_mod(R, R, P); // s^2 - 2x = xr

    sub_mod(t, R, P); // xr - x
    mult_mod(R+6, s, t); // s(xr - x)
    add_mod(R+6, R+6, P+6); // s(xr - x) - y
    neg_mod(R+6, R+6); // y' = -(s(xr - x) - y)
}

void add_P(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
    uint64_t s[6], t[6];

    if(is_zero(P) && is_zero(P+6)){
        copy_P(R, Q);
        return;
    }

    else if(is_zero(Q) && is_zero(Q+6)){
        copy_P(R, P);
        return;
    }

    else if(eq(P, Q)){
        if(eq(P+6, Q+6)){
            double_P(R, P);
        } else {
            printf("p = -q\n");
            memset(R, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    sub_mod(R, Q, P); // xq - xp
    inverse_mod(R+6, R); // 1/(xq - xp)

    sub_mod(R, Q+6, P+6); // yq - yp

    mult_mod(s, R, R+6); // s = (yq - yp)/(xq - xp)

    mult_mod(R, s, s); // s^2
    sub_mod(R, R, P); // s^2 - xp
    sub_mod(R, R, Q); // s^2 - xp - xq = xr

    sub_mod(t, R, P); // xr - xp
    mult_mod(R+6, s, t); // s(xr - xp)
    add_mod(R+6, R+6, P+6); // s(xr - xp) + yp
    neg_mod(R+6, R+6); // y' = -(s(xr - xp) + yp)
}

void neg_P(uint64_t R[12], uint64_t P[12]){
    copy(R, P);
    neg_mod(R+6, P+6);
}

void scalar_P(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
    if(is_zero_P(P) || is_zero(k)){
        memset(R, 0, 12*sizeof(uint64_t));
        return;
    }

    uint64_t Q[12], tQ[12], tR[12];
    memset(R, 0, 12*sizeof(uint64_t));
    copy_P(Q, P);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            add_P(tR, R, Q);
            copy_P(R, tR);
        }

        double_P(tQ, Q);
        copy_P(Q, tQ);
    }
}


void to_mon_P(uint64_t R[12], uint64_t P[12]){
    to_mon(R, P);
    to_mon(R+6, P+6);
}

void from_mon_P(uint64_t R[12], uint64_t P[12]){
    from_mon(R, P);
    from_mon(R+6, P+6);
}

void double_mon_P(uint64_t R[12], uint64_t P[12]){
    uint64_t s[6], t[6];

    add_mod(R, P+6, P+6); // 2y
    inverse_mon(t, R);  // 1/2y <- s
    to_mon(R+6, t);

    mult_mon(R, P, P); // x^2
    add_mod(s, R, R); // 2x^2
    add_mod(R, R, s); // 3x^2
    
    mult_mon(s, R, R+6); // 3x^2/2y
    mult_mon(R, s, s); // s^2

    sub_mod(R, R, P); // s^2 - x
    sub_mod(R, R, P); // s^2 - 2x = xr

    sub_mod(t, R, P); // xr - x
    mult_mon(R+6, s, t); // s(xr - x)
    add_mod(R+6, R+6, P+6); // s(xr - x) - y
    neg_mod(R+6, R+6); // y' = -(s(xr - x) - y)
}

void add_mon_P(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
    uint64_t s[6], t[6];

    if(is_zero(P) && is_zero(P+6)){
        copy_P(R, Q);
        return;
    }

    else if(is_zero(Q) && is_zero(Q+6)){
        copy_P(R, P);
        return;
    }

    else if(eq(P, Q)){
        if(eq(P+6, Q+6)){
            double_mon_P(R, P);
        } else {
            printf("p = -q\n");
            memset(R, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    sub_mod(R, Q, P); // xq - xp
    inverse_mon(t, R);  // 1/2y <- s
    to_mon(R+6, t);

    sub_mod(R, Q+6, P+6); // yq - yp

    mult_mon(s, R, R+6); // s = (yq - yp)/(xq - xp)

    mult_mon(R, s, s); // s^2
    sub_mod(R, R, P); // s^2 - xp
    sub_mod(R, R, Q); // s^2 - xp - xq = xr

    sub_mod(t, R, P); // xr - xp
    mult_mon(R+6, s, t); // s(xr - xp)
    add_mod(R+6, R+6, P+6); // s(xr - xp) + yp
    neg_mod(R+6, R+6); // y' = -(s(xr - xp) + yp)
}

void scalar_mon_P(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
    uint64_t Q[12], tQ[12], tR[12];
    memset(R, 0, 12*sizeof(uint64_t));
    copy_P(Q, P);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            add_mon_P(tR, R, Q);
            copy_P(R, tR);
        }
        double_mon_P(tQ, Q);
        copy_P(Q, tQ);
    }
}


void trivial_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(R, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        memset(T1, 0, 12*sizeof(uint64_t));
        scalar_P(T1, P + 12*i, k + 6*i);
        add_P(T2, R, T1);
        copy_P(R, T2);
    }
}

void mon_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(R, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        to_mon_P(T1, P + 12*i);
        scalar_mon_P(T2, T1, k + 6*i);
        add_mon_P(T1, R, T2);
        copy_P(R, T1);
    }
    from_mon_P(T1, R);
    copy_P(R, T1);
}

void bucket_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n){
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
                add_P(temp, buckets[index], P + 12*i);
                copy_P(buckets[index], temp);
            }
            
            memset(temp_sum, 0, 12*sizeof(uint64_t));
            memset(temp, 0, 12*sizeof(uint64_t));

            for(uint64_t i = 65535; i >= 1; i--){
                add_P(temp, temp_sum, buckets[i]);
                copy_P(temp_sum, temp);

                add_P(temp, bucket_sum, temp_sum);
                copy_P(bucket_sum, temp);
            }
            uint64_t shift_mul[6] = {0, 0, 0, 0, 0, 0,};
            shift_mul[w] = 1lu << shift;
            scalar_P(temp, bucket_sum, shift_mul);
            add_P(temp_sum, R, temp);
            copy_P(R, temp_sum);
        }
    }
}