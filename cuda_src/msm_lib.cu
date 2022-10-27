#include "msm_lib.hu"

__const__ uint64_t One[6] = {
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};
__const__ uint64_t P[6] = {
    0x8508c00000000001,
    0x170b5d4430000000,
    0x1ef3622fba094800,
    0x1a22d9f300f5138f,
    0xc63b05c06ca1493b,
    0x01ae3a4617c510ea
};
__const__ uint64_t R[6] = {
    0x7af73fffffffffff,
    0xe8f4a2bbcfffffff,
    0xe10c9dd045f6b7ff,
    0xe5dd260cff0aec70,
    0x39c4fa3f935eb6c4,
    0x0051c5b9e83aef15
};
__const__ uint64_t Ri[6] = {
    0x2b0909a28934f3a1,
    0x83264aa55c1cfac6,
    0x1accd49ca2a491ae,
    0xa28b2dce9e80e9a6,
    0x34d313ea126f7c08,
    0x0161de1ee3625456
};
__const__ uint64_t R2[6] = {
    0x30832a73a1b25004,
    0xa404dcf0bcb14011,
    0xb7520b89a32a1bcc,
    0x6154e2a7cfb8d35a,
    0x475fc7349417d690,
    0x0155f398d8e0e30f
};
__const__ uint64_t G1[12] = {
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


//// 1. UINT-384 Arithmetic ////

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

__host__ __device__ bool is_zero(uint64_t a[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != 0) {
            return false;
        }
    }
    return true;
}
__host__ __device__ bool geq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return true;
}
__host__ __device__ bool eq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__host__ __device__ uint64_t leading_zeros(uint64_t a[6]) {
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

__host__ __device__ void copy(uint64_t res[6], uint64_t a[6]){
    memcpy(res, a, 6 * sizeof(uint64_t));
}

__host__ __device__ void rshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 0; i < 6; i++) {
        res[i] = (a[i] >> 1);
        if (i < 5) res[i] |= (a[i+1] << (63));
    }
}
__host__ __device__ void lshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 5; i >= 0; i--) {
        res[i] = (a[i] << 1);
        if (i > 0) res[i] |= (a[i-1] >> (63));
    }
}
__host__ __device__ bool add(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] + b[i] + carry;
        res[i] = sum;
        carry = sum < ai;
    }
    return carry;
}
__host__ __device__ bool sub(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] - b[i] - carry;
        res[i] = sum;
        carry = sum > ai;
    }
    return carry;
}


//// 2. 377bit Field Arithmethic ////

void random_mod(uint64_t res[6]){
    for(int i = 0; i < 5; i++) res[i] = ((uint64_t)rand()<<32) | rand();
    res[5] = (((uint64_t)rand()<<32) | rand()) & 0x01ae3a4617c510ea;
    // while(geq(res, P)) {
    //     ((uint64_t)rand()<<32) | rand() & 0x01ffffffffffffff;
    // }
}

__host__ __device__ void add_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    add(res, a, b);
    if(geq(res, (uint64_t*)P)) {
        sub(res, res, (uint64_t*)P);
    }
}
__host__ __device__ void double_mod(uint64_t res[6], uint64_t a[6]) {
    lshift(res, a);
    if(geq(res, (uint64_t*)P)) {
        sub(res, res, (uint64_t*)P);
    }
}
__host__ __device__ void sub_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    if(geq(a, b)) {
        sub(res, a, b);
    } else {
        add(res, a, (uint64_t*)P);
        sub(res, res, b);
    }
}
__host__ __device__ void neg_mod(uint64_t res[6], uint64_t a[6]) {
    if(is_zero(a)) {
        copy(res, a);
    } else {
        sub(res, (uint64_t*)P, a);
    }
}
__host__ __device__ void mult_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
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
__host__ __device__ void inverse_mod(uint64_t res[6], uint64_t a[6]){
    uint64_t r[6] = {0,0,0,0,0,0};
    inverse_mon(r, a);
    mult_mon(res, r, (uint64_t*)One);
}

__host__ __device__ void to_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)R2);
}
__host__ __device__ void from_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)One);
}

__host__ __device__ void mult_mon(uint64_t res[6], uint64_t a[6], uint64_t b[6]){
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
            add(res, res, (uint64_t*)P);
        }
        rshift(res, res);
    }
    if(geq(res, (uint64_t*)P)){
        sub(res, res, (uint64_t*)P);
    }
}
__host__ __device__ void inverse_mon(uint64_t res[6], uint64_t a[6]){
    uint64_t u[6];
    uint64_t v[6];
    uint64_t s[6] = {1,0,0,0,0,0};
    uint64_t r[6] = {0,0,0,0,0,0};
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


//// 3. Elliptic Curve Points Arithmetic ////

void print_p(uint64_t p[12]){
    print(p);
    print(p+6);
}
void random_p(uint64_t res[12]){
    uint64_t k[6];
    random_mod(k);
    scalar_p(res, (uint64_t*)G1, k);
}

__host__ __device__ bool is_zero_p(uint64_t p[12]){
    return is_zero(p) && is_zero(p+6);
}
__host__ __device__ bool is_p(uint64_t p[12]){
    if(is_zero_p(p)) return true;
    uint64_t y2[6];
    mult_mod(y2, p+6, p+6);
    uint64_t x2[6], x3[6];
    mult_mod(x2, p, p);
    mult_mod(x3, x2, p);
    add_mod(x3, x3, (uint64_t*)One);
    return eq(x3, y2);
}
__host__ __device__ bool eq_p(uint64_t p[12], uint64_t q[12]){
    return eq(p, q) && eq(p+6, q+6);
}

__host__ __device__ void copy_p(uint64_t res[12], uint64_t p[12]){
    for(int i = 0; i < 12; i++) res[i] = p[i];
}
__host__ __device__ void copy_G1(uint64_t res[12]){
    copy_p(res, (uint64_t*)G1);
}

__host__ __device__ void double_p(uint64_t res[12], uint64_t p[12]){
    uint64_t s[6], t[6];

    add_mod(res, p+6, p+6); // 2y
    inverse_mod(res+6, res);  // 1/2y <- 

    mult_mod(res, p, p); // x^2
    add_mod(s, res, res); // 2x^2
    add_mod(res, res, s); // 3x^2
    
    mult_mod(s, res, res+6); // 3x^2/2y
    mult_mod(res, s, s); // s^2

    sub_mod(res, res, p); // s^2 - x
    sub_mod(res, res, p); // s^2 - 2x = xr

    sub_mod(t, res, p); // xr - x
    mult_mod(res+6, s, t); // s(xr - x)
    add_mod(res+6, res+6, p+6); // s(xr - x) - y
    neg_mod(res+6, res+6); // y' = -(s(xr - x) - y)
}
__host__ __device__ void add_p(uint64_t res[12], uint64_t p[12], uint64_t q[12]){
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
            double_p(res, p);
        } else {
            printf("p = -q\n");
            memset(res, 0, 12*sizeof(uint64_t));
        }
        return;
    }

    sub_mod(res, q, p); // xq - xp
    inverse_mod(res+6, res); // 1/(xq - xp)

    sub_mod(res, q+6, p+6); // yq - yp

    mult_mod(s, res, res+6); // s = (yq - yp)/(xq - xp)

    mult_mod(res, s, s); // s^2
    sub_mod(res, res, p); // s^2 - xp
    sub_mod(res, res, q); // s^2 - xp - xq = xr

    sub_mod(t, res, p); // xr - xp
    mult_mod(res+6, s, t); // s(xr - xp)
    add_mod(res+6, res+6, p+6); // s(xr - xp) + yp
    neg_mod(res+6, res+6); // y' = -(s(xr - xp) + yp)
}
__host__ __device__ void neg_p(uint64_t res[12], uint64_t p[12]){
    copy(res, p);
    neg_mod(res+6, p+6);
}
__host__ __device__ void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k[6]){
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
__host__ __device__ void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k){
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

__host__ __device__ void to_mon_p(uint64_t res[12], uint64_t p[12]){
    to_mon(res, p);
    to_mon(res+6, p+6);
}
__host__ __device__ void from_mon_p(uint64_t res[12], uint64_t p[12]){
    from_mon(res, p);
    from_mon(res+6, p+6);
}

__host__ __device__ void double_mon_p(uint64_t res[12], uint64_t p[12]){
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
__host__ __device__ void add_mon_p(uint64_t res[12], uint64_t p[12], uint64_t q[12]){
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
__host__ __device__ void scalar_mon_p(uint64_t res[12], uint64_t p[12], uint64_t k[6]){
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


//// 4. MSM Algorithms ////

void trivial_msm(uint64_t res[12], uint64_t *ps, uint64_t *ks, uint64_t n){
    uint64_t temp1[12], temp2[12];
    memset(res, 0, 12*sizeof(uint64_t));
    for(uint64_t i = 0; i < n; i++){
        memset(temp1, 0, 12*sizeof(uint64_t));
        scalar_p(temp1, ps + 12*i, ks + 6*i);
        add_p(temp2, res, temp1);
        copy_p(res, temp2);
    }
}

__global__ void map_to_mon(uint64_t*__restrict ress, uint64_t*__restrict ps, uint64_t n){
    uint64_t p[12], r[12];
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        copy_p(p, ps + 12*i);
        to_mon_p(r, p);
        copy_p(ress + 12*i, r);
    }
}
__global__ void map_mon_scalar(uint64_t*__restrict ress, uint64_t*__restrict ps, uint64_t*__restrict ks, uint64_t n){
    uint64_t p[12], k[6], r[12];
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        copy_p(p, ps + 12*i);
        copy(k, ks + 6*i);
        scalar_mon_p(r, p, k);
        copy_p(ress + 12*i, r);
    }
}
__global__ void linear_mon_reduce(uint64_t*__restrict ress, uint64_t*__restrict ps, uint64_t n){
    uint64_t p[12];
    uint64_t r[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t per_thread = n / (gridDim.x * blockDim.x); // points per thread
    uint64_t residue = n % (gridDim.x * blockDim.x); // residue points
    uint64_t pid = tid * per_thread; // point id
    
    if (tid < residue) {
        per_thread += 1;
        pid += tid;
    }
    else {
        pid += residue;
    }

    uint64_t stop = pid + per_thread;

    while(pid < n && pid < stop){
        copy_p(p, r);
        add_mon_p(r, p, ps + 12*pid);
        pid++;
    }

    copy_p(ress + 12*tid, r);
}
__global__ void log_mon_reduce(uint64_t*__restrict ress, uint64_t*__restrict ps, uint64_t n){

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t p1_id = 2*tid;
    uint64_t p2_id = 2*tid + 1;

    uint64_t r[12], p1[12], p2[12] = {0};

    if(p1_id < n){
        copy_p(p1, ps + 12*p1_id);
        if (p2_id < n) copy_p(p2, ps + 12*p2_id);
        add_mon_p(r, p1, p2);
        copy_p(ress + 12*tid, r);
    } 
}

void cuda_mon_msm(uint64_t __restrict res[12], uint64_t* __restrict ps, uint64_t* __restrict ks, uint64_t n){
    uint64_t k_size = n * 6 * sizeof(uint64_t);
    uint64_t p_size = n * 12 * sizeof(uint64_t);
    uint64_t r_size = n * 12 * sizeof(uint64_t);

    uint64_t *rs = (uint64_t*)malloc(r_size);

    uint64_t tp_block = 256;
    uint64_t scalar_blocks = (n + tp_block - 1) / tp_block;
    uint64_t reduce_blocks = 10;

    uint64_t *ks_dev;
    uint64_t *ps_dev;
    uint64_t *rs_dev;

    cudaMalloc((void**)&ks_dev, k_size);
    cudaMalloc((void**)&ps_dev, p_size);
    cudaMalloc((void**)&rs_dev, r_size);

    cudaMemcpy(ks_dev, ks, k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ps_dev, ps, p_size, cudaMemcpyHostToDevice);

    map_to_mon<<<scalar_blocks, tp_block>>>(rs_dev, ps_dev, n);
    map_mon_scalar<<<scalar_blocks, tp_block>>>(ps_dev, rs_dev, ks_dev, n);
    linear_mon_reduce<<<reduce_blocks, tp_block>>>(rs_dev, ps_dev, n);
    cudaMemcpy(ps_dev, rs_dev, p_size, cudaMemcpyDeviceToDevice);

    uint64_t elements = tp_block * reduce_blocks; // 2560
    while(elements > 1){
        uint64_t dr_blocks = (elements + tp_block - 1) / tp_block;
        uint64_t dr_tpb = (elements + dr_blocks - 1) / dr_blocks;

        log_mon_reduce<<<dr_blocks, dr_tpb>>>(rs_dev, ps_dev, elements);
        cudaMemcpy(ps_dev, rs_dev, p_size, cudaMemcpyDeviceToDevice);
        elements = (elements + 1) / 2;
    }
    cudaMemcpy(rs, rs_dev, 12*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    from_mon_p(res, rs);
}
