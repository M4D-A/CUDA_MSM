#include "cuda_src/msm_lib.hu"
#include "chrono"

int main(){
    uint64_t data_num = 2560 * 8;
    uint64_t k_size = data_num * 6 * sizeof(uint64_t);
    uint64_t P_size = data_num * 12 * sizeof(uint64_t);

    uint64_t *k_host = (uint64_t*)malloc(k_size);
    uint64_t *P_host = (uint64_t*)malloc(P_size);
    uint64_t R[12], cR[12];
    memset(k_host, 0, k_size);

    auto gen_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < data_num; i++){
        random_mod(k_host + i * 6);
        copy_P(P_host + i * 12, (uint64_t*)G1);
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_diff = gen_end - gen_start;
    printf("Generate time: %f\n", gen_diff.count());

    auto tmsm_start = std::chrono::high_resolution_clock::now();
    trivial_msm(R, P_host, k_host, data_num);
    auto tmsm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tmsm_diff = tmsm_end - tmsm_start;
    print_P(R);
    printf("%d %lf\n",is_P(R), tmsm_diff.count());

    auto cmsm_start = std::chrono::high_resolution_clock::now();
    cuda_msm(cR, P_host, k_host, data_num);
    auto cmsm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cmsm_diff = cmsm_end - cmsm_start;
    print_P(cR);
    printf("%d %lf\n",is_P(cR), cmsm_diff.count());

}