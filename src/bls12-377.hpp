#include "uint384.hpp"
#include "field377.hpp"

extern uint64_t G1[12];

void printP(uint64_t P[12]);
bool isPoint(uint64_t P[12]);
bool isZeroP(uint64_t P[12]);
void randomP(uint64_t res[12]);
void copyP(uint64_t res[12], uint64_t P[12]);
void doubleP(uint64_t res[12], uint64_t P[12]);
void addP(uint64_t res[12], uint64_t P[12], uint64_t q[12]);
void negP(uint64_t res[12], uint64_t P[12]);
void scalarP(uint64_t res[12], uint64_t P[12], uint64_t k[6]);
void msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);


void toMonP(uint64_t R[12], uint64_t P[12]);
void fromMonP(uint64_t R[12], uint64_t P[12]);
void doublePMon(uint64_t R[12], uint64_t P[12]);
void addPMon(uint64_t R[12], uint64_t P[12], uint64_t Q[12]);
void scalarPMon(uint64_t R[12], uint64_t P[12], uint64_t k[6]);
void msmMon(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);

void msm2(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);
void bucketMSM(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);