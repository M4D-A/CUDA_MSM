#include "uint384.hpp"
#include "field377.hpp"

void print_p(uint64_t p[12]);
void random_p(uint64_t res[12]);

bool is_zero_p(uint64_t p[12]);
bool is_p(uint64_t p[12]);
bool eq_p(uint64_t p[12], uint64_t q[12]);

void copy_p(uint64_t res[12], uint64_t p[12]);
void copy_G1(uint64_t res[12]);

void double_p(uint64_t res[12], uint64_t p[12]);
void add_p(uint64_t res[12], uint64_t p[12], uint64_t q[12]);
void neg_p(uint64_t res[12], uint64_t p[12]);
void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k[6]);
void scalar_p(uint64_t res[12], uint64_t p[12], uint64_t k);

void to_mon_p(uint64_t r[12], uint64_t p[12]);
void from_mon_p(uint64_t r[12], uint64_t p[12]);
void double_mon_p(uint64_t r[12], uint64_t p[12]);
void add_mon_p(uint64_t r[12], uint64_t p[12], uint64_t q[12]);
void scalar_mon_p(uint64_t r[12], uint64_t p[12], uint64_t k[6]);

void trivial_msm(uint64_t r[12], uint64_t *p, uint64_t *k, uint64_t n);
void mon_msm(uint64_t r[12], uint64_t *p, uint64_t *k, uint64_t n);
void bucket_msm(uint64_t r[12], uint64_t *p, uint64_t *k, uint64_t n);