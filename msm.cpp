#include <inttypes.h>
#include "src/uint384.hpp"
#include "src/field377.hpp"
#include "src/bls12-377.hpp"

int main(){
    uint n = 10;

    uint64_t *k = new uint64_t[6 * n];
    uint64_t *P = new uint64_t[12 * n];
    uint64_t res1[12], res2[12], res3[12];

    for(int i =0 ; i < n; i++){
        randomMod(k + 6 * i);
        randomP(P + 12 * i);
    }

    for(int i =0 ; i < n; i++){
        printf("%d \n", i);
        print(k + 6 * i);
        printP(P + 12 * i);
    }

    printf("\n\n");

    msm(res1, P, k, n);
    printP(res1);
    printf("%d\n", isPoint(res1));
    
    msm2(res2, P, k, n);
    printP(res2);
    printf("%d\n", isPoint(res2));

    bucketMSM(res3, P, k, n);
    printP(res3);
    printf("%d\n", isPoint(res3));
}