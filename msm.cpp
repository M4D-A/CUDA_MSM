#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

void m_add(uint32_t *A, uint32_t *B, uint32_t *C, uint32_t n) { // n-32 X n-32 -> (n+1)-32
  uint32_t residue = 0;
  uint32_t mask32 = 0xFFFFFFFF;
  for (int64_t i = 0; i < n; i++) {
    uint64_t sum = (uint64_t)A[i] + (uint64_t)B[i] + (uint64_t)residue;
    residue = sum >> 32;
    C[i] = sum & mask32;
  }
  C[n] += residue;
}

void m_sub(uint32_t *A, uint32_t *B, uint32_t *C, uint32_t n) { // n-32 X n-32 -> (n+1)-32
  uint32_t residue = 0;
  uint32_t mask32 = 0xFFFFFFFF;
  for (int64_t i = 0; i < n; i++) {
    if(A[i] - residue >= B[i]) {
        C[i] = A[i] - residue - B[i];
        residue = 0;

    } else {
        C[i] = A[i] - residue - B[i];
        residue = 1;
    }
  }
  C[n] += residue;
}

void t3_mult(uint32_t A[3], uint32_t B[3], uint32_t C[6]){
    uint32_t mask = 0xFFFFFFFF;

    int64_t pt = A[0] + A[2];
    int64_t p0 = A[0];
    int64_t p1 = pt + A[1];
    int64_t pn1 = pt - A[1];
    int64_t pn2 = (pn1 + A[2]) * 2 - A[0];
    int64_t pinf = A[2];

    int64_t qt = B[0] + B[2];
    int64_t q0 = B[0];
    int64_t q1 = qt + B[1];
    int64_t qn1 = qt - B[1];
    int64_t qn2 = (qn1 + B[2]) * 2 - B[0];
    int64_t qinf = B[2];

    int64_t r0 = p0 * q0;
    int64_t r1 = p1 * q1;
    int64_t rn1 = pn1 * qn1;
    int64_t rn2 = pn2 * qn2;
    int64_t rinf = pinf * qinf;

    int64_t c0 = r0;
    int64_t c4 = rinf;
    int64_t c3 = (rn2 - r1)/3;
    int64_t c1 = (r1 - rn1)/2;
    int64_t c2 = (rn1 - r0);
    c3 = (c2-c3)/2 + 2*rinf;
    c2 = c2 + c1 - c4;
    c1 = c1 - c3;

    C[0] = c0 & mask;
    C[1] = c1 & mask + (c0 / (1lu << 32));
    C[2] = c2 & mask + (c1 / (1lu << 32));
    C[3] = c3 & mask + (c2 / (1lu << 32));
    C[4] = c4 & mask + (c3 / (1lu << 32));
    C[5] = (c4 / (1lu << 32));
}

void t6_mult(uint32_t A[6], uint32_t B[6], uint32_t C[12]){
    uint32_t Z0[6];
    uint32_t Z2[6];
    t3_mult(A, B, Z0);
    t3_mult(A+3, B+3, Z2);

    printf("Z0: ");
    for(int i = 0; i < 6; i++){
        printf("%08X ", Z0[5 - i]);
    }
    printf("\nZ2: ");
    for(int i = 0; i < 6; i++){
        printf("%08X ", Z2[5 - i]);
    }
    printf("\n");

    uint32_t AT[4];
    uint32_t BT[4];
    m_add(A, A+3, AT, 3);
    m_add(B, B+3, BT, 3);

    uint32_t Z1[7];

    t3_mult(AT, BT, Z1);
    if(AT[3] == 1){
        m_add(Z1+3, BT, Z1+3, 3);
    }
    if(BT[3] == 1){
        m_add(Z1+3, AT, Z1+3, 3);
    }
    if(AT[3] == 1 && BT[3] == 1){
        Z1[6] += 1;
    }

    m_sub(Z1, Z0, Z1, 6);
    m_sub(Z1, Z2, Z1, 6);





    for(int i = 0; i < 6; i++){
        C[i] = Z0[i];
        C[i+6] = Z2[i];
    }

    m_add(Z1, C+3, C+3, 6);
}

int main(){
    uint32_t A[6] = {0x456789AB, 0x56789ABC, 0x6789ABCD};
    uint32_t B[6] = {0x456789AB, 0x56789ABC, 0x6789ABCD};
    uint32_t C[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    t3_mult(A, B, C);
    for(int i = 0; i < 12; i++){
        printf("%08x ", C[11-i]);
    }
}