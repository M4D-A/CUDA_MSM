#include "uint384.hpp"
extern uint64_t P[6];
extern uint64_t R[6];
extern uint64_t Ri[6];
extern uint64_t R2[6];

void random_mod(uint64_t res[6]);
void add_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void sub_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void neg_mod(uint64_t res[6], uint64_t a[6]);
void mult_mod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void mult_mon(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void to_mon(uint64_t res[6], uint64_t a[6]);
void from_mon(uint64_t res[6], uint64_t a[6]);
void inverse_mon(uint64_t res[6], uint64_t a[6]);
void inverse_mod(uint64_t res[6], uint64_t a[6]);