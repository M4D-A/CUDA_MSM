#include "uint384.hpp"
#include "field377.hpp"
#include "bls12-377.hpp"
#include <chrono>

int main(){
    uint64_t data_num = 256;

    uint64_t k_host[ 6 * data_num];
    uint64_t P_host[12 * data_num];
    uint64_t R[12];

    auto gen_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < data_num; i++){
        random_mod(k_host + i * 6);
        copy_G1(P_host + i * 12);
    }
    auto gen_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_diff = gen_end - gen_start;
    printf("Generate time: %f\n", gen_diff.count());

    auto tmsm_start = std::chrono::high_resolution_clock::now();
    trivial_msm(R, P_host, k_host, data_num);
    auto tmsm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tmsm_diff = tmsm_end - tmsm_start;
    print_p(R);
    printf("%lf\n", tmsm_diff.count());

    auto mmsm_start = std::chrono::high_resolution_clock::now();
    mon_msm(R, P_host, k_host, data_num);
    auto mmsm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mmsm_diff = mmsm_end - mmsm_start;
    print_p(R);
    printf("%lf\n", mmsm_diff.count());
}