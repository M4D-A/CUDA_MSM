#include "bls12-377.hpp"

const uint64_t One[6] = {
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};
const uint64_t G1[12] = {
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

void print_p(uint64_t p[12]){
    print(p);
    print(p+6);
    printf("%d\n", is_p(p));
}
void random_p(uint64_t res[12]){
    uint64_t k[6];
    random_mod(k);
    scalar_p(res, (uint64_t*)G1, k);
}

bool is_zero_p(uint64_t p[12]){
    return is_zero(p) && is_zero(p+6);
}
bool is_p(uint64_t p[12]){
    if(is_zero_p(p)) return true;
    uint64_t y2[6];
    mult_mod(y2, p+6, p+6);
    uint64_t x2[6], x3[6];
    mult_mod(x2, p, p);
    mult_mod(x3, x2, p);
    add_mod(x3, x3, (uint64_t*)One);
    return eq(x3, y2);
}
bool eq_p(uint64_t p[12], uint64_t q[12]){
    return eq(p, q) && eq(p+6, q+6);
}

void copy_p(uint64_t res[12], uint64_t p[12]){
    memcpy(res, p, 12*sizeof(uint64_t));
}
void copy_G1(uint64_t res[12]){
    copy_p(res, (uint64_t*)G1);
}

void double_p(uint64_t res[12], uint64_t p[12]){
    uint64_t s[6], t[6];

    add_mod(res, p + 6, p + 6); // 2y
    inverse_mod(res + 6, res);  // 1/2y <- 

    mult_mod(res, p, p); // x^2
    add_mod(s, res, res); // 2x^2
    add_mod(res, res, s); // 3x^2
    
    mult_mod(s, res, res + 6); // 3x^2/2y
    mult_mod(res, s, s); // s^2

    sub_mod(res, res, p); // s^2 - x
    sub_mod(res, res, p); // s^2 - 2x = xr

    sub_mod(t, res, p); // xr - x
    mult_mod(res + 6, s, t); // s(xr - x)
    add_mod(res + 6, res + 6, p + 6); // s(xr - x) - y
    neg_mod(res + 6, res + 6); // y' = -(s(xr - x) - y)
}
void add_p(uint64_t res[12], uint64_t p[12], uint64_t q[12]){
    uint64_t s[6], t[6];

    if(is_zero(p) && is_zero(p + 6)){
        copy_p(res, q);
        return;
    }

    else if(is_zero(q) && is_zero(q + 6)){
        copy_p(res, p);
        return;
    }

    else if(eq(p, q)){
        if(eq(p + 6, q + 6)){
            double_p(res, p);
        } else {
            printf("p = -q\n");
            memset(res, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    sub_mod(res, q, p); // xq - xp
    inverse_mod(res + 6, res); // 1/(xq - xp)

    sub_mod(res, q + 6, p + 6); // yq - yp

    mult_mod(s, res, res + 6); // s = (yq - yp)/(xq - xp)

    mult_mod(res, s, s); // s^2
    sub_mod(res, res, p); // s^2 - xp
    sub_mod(res, res, q); // s^2 - xp - xq = xr

    sub_mod(t, res, p); // xr - xp
    mult_mod(res + 6, s, t); // s(xr - xp)
    add_mod(res + 6, res + 6, p + 6); // s(xr - xp)  +  yp
    neg_mod(res + 6, res + 6); // y' = -(s(xr - xp)  +  yp)
}
void neg_p(uint64_t res[12], uint64_t p[12]){
    copy(res, p);
    neg_mod(res + 6, p + 6);
}
void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k[6]){
    if(is_zero_p(p) || is_zero(k)){
        memset(res, 0, 12*sizeof(uint64_t));
        return;
    }

    uint64_t q[12], tq[12], tres[12];
    memset(res, 0, 12*sizeof(uint64_t));
    copy_p(q, p);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            add_p(tres, res, q);
            copy_p(res, tres);
        }

        double_p(tq, q);
        copy_p(q, tq);
    }
}
void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k){
    if(is_zero_p(p) || k == 0){
        memset(res, 0, 12*sizeof(uint64_t));
        return;
    }

    uint64_t q[12], tq[12], tres[12];
    memset(res, 0, 12*sizeof(uint64_t));
    copy_p(q, p);
    for(int i = 0; i < 64; i++){
        if((k >> i) & 1u){
            add_p(tres, res, q);
            copy_p(res, tres);
        }

        double_p(tq, q);
        copy_p(q, tq);
    }
}

void to_mon_p(uint64_t res[12], uint64_t p[12]){
    to_mon(res, p);
    to_mon(res+6, p+6);
}
void from_mon_p(uint64_t res[12], uint64_t p[12]){
    from_mon(res, p);
    from_mon(res+6, p+6);
}
void double_mon_p(uint64_t res[12], uint64_t p[12]){
    uint64_t s[6], t[6];

    add_mod(res, p+6, p+6); // 2y
    inverse_mon(t, res);  // 1/2y <- s
    to_mon(res+6, t);

    mult_mon(res, p, p); // x^2
    add_mod(s, res, res); // 2x^2
    add_mod(res, res, s); // 3x^2
    
    mult_mon(s, res, res+6); // 3x^2/2y
    mult_mon(res, s, s); // s^2

    sub_mod(res, res, p); // s^2 - x
    sub_mod(res, res, p); // s^2 - 2x = xr

    sub_mod(t, res, p); // xr - x
    mult_mon(res+6, s, t); // s(xr - x)
    add_mod(res+6, res+6, p+6); // s(xr - x) - y
    neg_mod(res+6, res+6); // y' = -(s(xr - x) - y)
}
void add_mon_p(uint64_t res[12], uint64_t p[12], uint64_t q[12]){
    uint64_t s[6], t[6];

    if(is_zero(p) && is_zero(p+6)){
        copy_p(res, q);
        return;
    }

    else if(is_zero(q) && is_zero(q+6)){
        copy_p(res, p);
        return;
    }

    else if(eq(p, q)){
        if(eq(p+6, q+6)){
            double_mon_p(res, p);
        } else {
            printf("p = -q\n");
            memset(res, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    sub_mod(res, q, p); // xq - xp
    inverse_mon(t, res);  // 1/2y <- s
    to_mon(res+6, t);

    sub_mod(res, q+6, p+6); // yq - yp

    mult_mon(s, res, res+6); // s = (yq - yp)/(xq - xp)

    mult_mon(res, s, s); // s^2
    sub_mod(res, res, p); // s^2 - xp
    sub_mod(res, res, q); // s^2 - xp - xq = xr

    sub_mod(t, res, p); // xr - xp
    mult_mon(res+6, s, t); // s(xr - xp)
    add_mod(res+6, res+6, p+6); // s(xr - xp) + yp
    neg_mod(res+6, res+6); // y' = -(s(xr - xp) + yp)
}
void scalar_mon_p(uint64_t res[12], uint64_t p[12], uint64_t k[6]){
    uint64_t q[12], tq[12], tres[12];
    memset(res, 0, 12*sizeof(uint64_t));
    copy_p(q, p);
    uint64_t bits = 384 - leading_zeros(k);
    for(int i = 0; i < bits; i++){
        int bit = i % 64;
        int word = i / 64;
        if(k[word] >> bit & 1){
            add_mon_p(tres, res, q);
            copy_p(res, tres);
        }
        double_mon_p(tq, q);
        copy_p(q, tq);
    }
}

void trivial_msm(uint64_t res[12], uint64_t *p, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(res, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        memset(T1, 0, 12*sizeof(uint64_t));        
        scalar_p(T1, p + 12*i, k + 6*i);
        add_p(T2, res, T1);
        copy_p(res, T2);
    }
}
void mon_msm(uint64_t res[12], uint64_t *p, uint64_t *k, uint64_t n){
    uint64_t T1[12], T2[12];
    memset(res, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        to_mon_p(T1, p + 12*i);
        scalar_mon_p(T2, T1, k + 6*i);
        add_mon_p(T1, res, T2);
        copy_p(res, T1);
    }
    from_mon_p(T1, res);
    copy_p(res, T1);
}
void bucket_msm(uint64_t res[12], uint64_t *p, uint64_t *k, uint64_t n){
    memset(res, 0, 12*sizeof(uint64_t));
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
                add_p(temp, buckets[index], p + 12*i);
                copy_p(buckets[index], temp);
            }
            
            memset(temp_sum, 0, 12*sizeof(uint64_t));
            memset(temp, 0, 12*sizeof(uint64_t));

            for(uint64_t i = 65535; i >= 1; i--){
                add_p(temp, temp_sum, buckets[i]);
                copy_p(temp_sum, temp);

                add_p(temp, bucket_sum, temp_sum);
                copy_p(bucket_sum, temp);
            }
            uint64_t shift_mul[6] = {0, 0, 0, 0, 0, 0,};
            shift_mul[w] = 1lu << shift;
            scalar_p(temp, bucket_sum, shift_mul);
            add_p(temp_sum, res, temp);
            copy_p(res, temp_sum);
        }
    }
}