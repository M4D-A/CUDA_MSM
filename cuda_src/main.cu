#include "msm_lib.hu"
#include "chrono"

int main(){
    uint64_t data_num = 2560 * 8;
    uint64_t k_size = data_num * 6 * sizeof(uint64_t);
    uint64_t P_size = data_num * 12 * sizeof(uint64_t);

    uint64_t *k_host = (uint64_t*)malloc(k_size);
    uint64_t *P_host = (uint64_t*)malloc(P_size);
    uint64_t R[12], cR[12], cmR[12];
    memset(k_host, 0, k_size);

    auto gen_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < data_num; i++){
        random_mod(k_host + i * 6);
        copy_G1(P_host + i * 12);
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_diff = gen_end - gen_start;
    printf("Generate time: %f\n", gen_diff.count());

    auto cmmsm_start = std::chrono::high_resolution_clock::now();
    cuda_mon_msm(cmR, P_host, k_host, data_num);
    auto cmmsm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cmmsm_diff = cmmsm_end - cmmsm_start;
    print_p(cmR);
    printf("cuda mon msm %d %lf\n",is_p(cmR), cmmsm_diff.count());

    // auto tmsm_start = std::chrono::high_resolution_clock::now();
    // trivial_msm(R, P_host, k_host, data_num);
    // auto tmsm_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> tmsm_diff = tmsm_end - tmsm_start;
    // print_p(R);
    // printf("triv msm %d %lf\n",is_p(R), tmsm_diff.count());
}