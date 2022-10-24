#include <inttypes.h>
#include "cuda_src/field377.hu"

int main(int argc, char **argv){
    uint64_t data_num = 4*256;
    uint64_t data_size = data_num * 12 * sizeof(uint64_t);

    uint64_t threads_per_block = 256;
    uint64_t blocks_per_grid = (data_num + threads_per_block - 1) / threads_per_block;

    uint64_t *data_a_host = (uint64_t*)malloc(data_size);
    uint64_t *data_b_host = (uint64_t*)malloc(data_size);
    uint64_t *data_c_host = (uint64_t*)malloc(data_size);

    for(int i = 0; i < data_num; i++){
        copy_P(data_a_host + i * 12, (uint64_t*)G1);
        copy_P(data_b_host + i * 12, (uint64_t*)G1);
    }

    uint64_t *data_a_dev;
    uint64_t *data_b_dev;
    uint64_t *data_c_dev;

    cudaMalloc((void**)&data_a_dev, data_size);
    cudaMalloc((void**)&data_c_dev, data_size);
    cudaMalloc((void**)&data_b_dev, data_size);

    cudaMemcpy(data_a_dev, data_a_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_dev, data_b_host, data_size, cudaMemcpyHostToDevice);

    add_P_kernel<<<blocks_per_grid, threads_per_block>>>(data_a_dev, data_b_dev, data_num, data_c_dev);

    cudaMemcpy(data_c_host, data_c_dev, data_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++){
        uint64_t c[12];
        add_P(c, &data_a_host[i*12], &data_b_host[i*12]);
        print_P(c);
        print_P(data_c_host + i*12);
        printf("%d\n", is_P(c));
        printf("%d\n", is_P(data_c_host + i*12));
        if(!eq(c, &data_c_host[i*12])){
            printf("Error at %d\n",i);
        }
    }

    return;
}