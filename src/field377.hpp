#include "uint384.hpp"
extern uint64_t P[6];
extern uint64_t R[6];
extern uint64_t Ri[6];
extern uint64_t R2[6];

void randomMod(uint64_t res[6]);
void addMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void subMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void negMod(uint64_t res[6], uint64_t a[6]);
void prodMod(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void prodMon(uint64_t res[6], uint64_t a[6], uint64_t b[6]);
void toMon(uint64_t res[6], uint64_t a[6]);
void fromMon(uint64_t res[6], uint64_t a[6]);
void inverseMon(uint64_t res[6], uint64_t a[6]);
void inverseMod(uint64_t res[6], uint64_t a[6]);