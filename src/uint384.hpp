#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdexcept>
#include <gtest/gtest.h>

extern uint64_t Zero[6];
extern uint64_t One[6];

void print(uint64_t a[6]);
bool geq(uint64_t a[6], uint64_t b[6]);
bool eq(uint64_t a[6], uint64_t b[6]);
bool is_zero(uint64_t a[6]);
uint64_t leading_zeros(uint64_t a[6]);
void copy(uint64_t res[6], uint64_t a[6]);
void rshift(uint64_t res[6], uint64_t a[6]);
void lshift(uint64_t res[6], uint64_t a[6]);
bool add(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
bool sub(uint64_t res[6], uint64_t a[6], uint64_t b[6]);