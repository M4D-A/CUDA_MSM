#include "uint384.hpp"
#include "field377.hpp"

extern uint64_t G1[12];

void print_P(uint64_t P[12]);
bool is_zero_P(uint64_t P[12]);
bool is_P(uint64_t P[12]);
void random_P(uint64_t res[12]);
void copy_P(uint64_t res[12], uint64_t P[12]);
void double_P(uint64_t res[12], uint64_t P[12]);
void add_P(uint64_t res[12], uint64_t P[12], uint64_t q[12]);
void neg_P(uint64_t res[12], uint64_t P[12]);
void scalar_P(uint64_t res[12], uint64_t P[12], uint64_t k[6]);

void to_mon_P(uint64_t R[12], uint64_t P[12]);
void from_mon_P(uint64_t R[12], uint64_t P[12]);
void double_mon_P(uint64_t R[12], uint64_t P[12]);
void add_mon_P(uint64_t R[12], uint64_t P[12], uint64_t Q[12]);
void scalar_mon_P(uint64_t R[12], uint64_t P[12], uint64_t k[6]);

void trivial_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);
void mon_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);
void bucket_msm(uint64_t R[12], uint64_t *P, uint64_t *k, uint64_t n);