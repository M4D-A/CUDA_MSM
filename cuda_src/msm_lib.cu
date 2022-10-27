#include "msm_lib.hu"


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

__host__ __device__ bool eq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
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

__host__ __device__ void to_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)R2);
}

__host__ __device__ void from_mon(uint64_t res[6], uint64_t a[6]){
    mult_mon(res, a, (uint64_t*)One);
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

__host__ __device__ void inverse_mod(uint64_t res[6], uint64_t a[6]){
    uint64_t r[6] = {0,0,0,0,0,0};
    inverse_mon(r, a);
    mult_mon(res, r, (uint64_t*)One);
}



//// 3. Elliptic Curve Points Arithmetic ////

void print_P(uint64_t P[12]){
    print(P);
    print(P+6);
}

void random_P(uint64_t res[12]){
    uint64_t k[6];
    random_mod(k);
    scalar_P(res, (uint64_t*)G1, k);
}


__host__ __device__ bool is_zero_P(uint64_t P[12]){
    return is_zero(P) && is_zero(P+6);
}

__host__ __device__ bool is_P(uint64_t P[12]){
    if(is_zero_P(P)) return true;
    uint64_t y2[6];
    mult_mod(y2, P+6, P+6);
    uint64_t x2[6], x3[6];
    mult_mod(x2, P, P);
    mult_mod(x3, x2, P);
    add_mod(x3, x3, (uint64_t*)One);
    return eq(x3, y2);
}

__host__ __device__ bool eq_P(uint64_t P[12], uint64_t Q[12]){
    return eq(P, Q) && eq(P+6, Q+6);
}

__host__ __device__ void copy_P(uint64_t R[12], uint64_t P[12]){
    for(int i = 0; i < 12; i++) R[i] = P[i];
}

__host__ __device__ void double_P(uint64_t R[12], uint64_t P[12]){
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

__host__ __device__ void add_P(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
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

__host__ __device__ void neg_P(uint64_t R[12], uint64_t P[12]){
    copy(R, P);
    neg_mod(R+6, P+6);
}

__host__ __device__ void scalar_P(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
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

__host__ __device__ void scalar_P(uint64_t R[12], uint64_t P[12], uint64_t k){
    if(is_zero_P(P) || k == 0){
        memset(R, 0, 12*sizeof(uint64_t));
        return;
    }

    uint64_t Q[12], tQ[12], tR[12];
    memset(R, 0, 12*sizeof(uint64_t));
    copy_P(Q, P);
    for(int i = 0; i < 64; i++){
        if((k >> i) & 1u){
            add_P(tR, R, Q);
            copy_P(R, tR);
        }

        double_P(tQ, Q);
        copy_P(Q, tQ);
    }
}

__host__ __device__ void to_mon_P(uint64_t R[12], uint64_t P[12]){
    to_mon(R, P);
    to_mon(R+6, P+6);
}

__host__ __device__ void from_mon_P(uint64_t R[12], uint64_t P[12]){
    from_mon(R, P);
    from_mon(R+6, P+6);
}

__host__ __device__ void double_mon_P(uint64_t R[12], uint64_t P[12]){
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

__host__ __device__ void add_mon_P(uint64_t R[12], uint64_t P[12], uint64_t Q[12]){
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

__host__ __device__ void scalar_mon_P(uint64_t R[12], uint64_t P[12], uint64_t k[6]){
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


//// 4. MSM Algorithms ////

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

__global__ void map_scalar(
    uint64_t* data_out,
    uint64_t* data_k,
    uint64_t* data_P,
    uint64_t  data_num){
    
    uint64_t P[12], k[6], R[12];
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < data_num){
        copy_P(P, data_P + 12*i);
        copy(k, data_k + 6*i);
        scalar_P(R, P, k);
        copy_P(data_out + 12*i, R);
    }
}

__global__ void linear_reduce(
    uint64_t* data_out,
    uint64_t* data_P,
    uint64_t  data_num){
    
    uint64_t P[12];
    uint64_t R[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t per_thread = data_num / (gridDim.x * blockDim.x); // points per thread
    uint64_t residue = data_num % (gridDim.x * blockDim.x); // residue points
    uint64_t pid = tid * per_thread; // point id
    
    if (tid < residue) {
        per_thread += 1;
        pid += tid;
    }
    else {
        pid += residue;
    }

    uint64_t stop = pid + per_thread;

    while(pid < data_num && pid < stop){
        copy_P(P, R);
        add_P(R, P, data_P + 12*pid);
        pid++;
    }

    copy_P(data_out + 12*tid, R);
}

__global__ void log_reduce(
    uint64_t* data_out,
    uint64_t* data_P,
    uint64_t  data_num){

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t p1 = 2*tid;
    uint64_t p2 = 2*tid + 1;

    uint64_t P1[12], P2[12], R[12];
    if(p1 < data_num){
        copy_P(P1, data_P + 12*p1);
        if(p2 < data_num){
            copy_P(P2, data_P + 12*p2);
            add_P(R, P1, P2);
        }
        else{
            copy_P(R, P1);
        }
        copy_P(data_out + 12*tid, R);
    }
}

void cuda_msm(uint64_t R[12], uint64_t *P_host, uint64_t *k_host, uint64_t n){
    uint64_t data_k_size = n * 6 * sizeof(uint64_t);
    uint64_t data_P_size = n * 12 * sizeof(uint64_t);
    uint64_t data_R_size = n * 12 * sizeof(uint64_t);

    uint64_t *R_host = (uint64_t*)malloc(data_R_size);

    uint64_t tp_block = 256;
    uint64_t scalar_blocks = (n + tp_block - 1) / tp_block;
    uint64_t reduce_blocks = 10;
    uint64_t threads = reduce_blocks*tp_block;

    uint64_t *data_k_dev;
    uint64_t *data_P_dev;
    uint64_t *data_R_dev;

    cudaMalloc((void**)&data_k_dev, data_k_size);
    cudaMalloc((void**)&data_P_dev, data_P_size);
    cudaMalloc((void**)&data_R_dev, data_R_size);

    cudaMemcpy(data_k_dev, k_host, data_k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(data_P_dev, P_host, data_P_size, cudaMemcpyHostToDevice);

    map_scalar<<<scalar_blocks, tp_block>>>(data_R_dev, data_k_dev, data_P_dev, n);
    cudaMemset(data_P_dev, 0, data_P_size);
    cudaMemcpy(R_host, data_R_dev, data_R_size, cudaMemcpyDeviceToHost);
    


    linear_reduce<<<reduce_blocks, tp_block>>>(data_P_dev, data_R_dev, n);
    cudaMemset(data_R_dev, 0, data_R_size);

    uint64_t elements = tp_block * reduce_blocks; // 2560
    while(elements > 1){
        uint64_t dr_blocks = (elements + tp_block - 1) / tp_block;
        uint64_t dr_tpb = (elements + dr_blocks - 1) / dr_blocks;

        log_reduce<<<dr_blocks, dr_tpb>>>(data_R_dev, data_P_dev, elements);
        cudaMemset(data_P_dev, 0, data_P_size);
        cudaMemcpy(data_P_dev, data_R_dev, data_P_size, cudaMemcpyDeviceToDevice);
        elements = (elements + 1) / 2;
    }

    cudaMemcpy(R, data_R_dev, 12*sizeof(uint64_t), cudaMemcpyDeviceToHost);
}