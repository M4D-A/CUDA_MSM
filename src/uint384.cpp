#include "uint384.hpp"

uint64_t Zero[6] = {
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};

uint64_t One[6] = {
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};

uint64_t random64(){
    uint64_t res = rand();
    res = res << 32;
    res = res | rand();
    return res;
}

void random384(uint64_t res[6]){
    for(int i = 0; i < 6; i++) res[i] = random64();
}

void print(uint64_t a[6]) {
    printf("0x");
    for (int i = 5; i >=0; i--) {
        printf("%016lX", a[i]);
        if(i != 0) printf("_");
    }
    printf("\n");
}

bool is_zero(uint64_t a[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != 0) {
            return false;
        }
    }
    return true;
}

bool eq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool geq(uint64_t a[6], uint64_t b[6]) {
    for (int i = 5; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return true;
}

uint64_t leading_zeros(uint64_t a[6]) {
    uint64_t result = 0;
    for (int i = 5; i >= 0; i--) {
        if (a[i] == 0) {
            result += 64;
        } else {
            result += __builtin_clzll(a[i]);
            break;
        }
    }
    return result;
}

void copy(uint64_t res[6], uint64_t a[6]){
    memcpy(res, a, 6 * sizeof(uint64_t));
}

void rshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 0; i < 6; i++) {
        res[i] = (a[i] >> 1);
        if (i < 5) res[i] |= (a[i+1] << (63));
    }
}

void lshift(uint64_t res[6], uint64_t a[6]){
    for (int i = 5; i >= 0; i--) {
        res[i] = (a[i] << 1);
        if (i > 0) res[i] |= (a[i-1] >> (63));
    }
}

bool add(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] + b[i] + carry;
        res[i] = sum;
        carry = sum < ai;
    }
    return carry;
}

bool sub(uint64_t res[6], uint64_t a[6], uint64_t b[6]) {
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t ai = a[i];
        uint64_t sum = a[i] - b[i] - carry;
        res[i] = sum;
        carry = sum > ai;
    }
    return carry;
}



#define test_runs 1000000

TEST(uint384, edge_test_is_zero) {
    uint64_t a[6] = {0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(is_zero(a));

    uint64_t b[6] = {0x1, 0x1, 0x1, 0x1, 0x1, 0x1};
    EXPECT_FALSE(is_zero(b));

    // one hot bit vectors
    for(uint i = 0; i<384; i++){
        uint word = i / 64;
        uint bit = i % 64;
        a[word] = 1 << bit;
        EXPECT_FALSE(is_zero(a));

        a[word] = 0;
        EXPECT_TRUE(is_zero(a));
    }
}

TEST(uint384, rand_test_is_zero) {
    uint64_t a[6] = {0, 0, 0, 0, 0, 0};
    for(int i = 0; i < test_runs; i++) {
        uint rand;
        do{
            rand = random64();
        }while (rand == 0);

        uint word = rand % 6;
        a[word] = rand;
        EXPECT_FALSE(is_zero(a));

        a[word] = 0;
        EXPECT_TRUE(is_zero(a));
    }
}


TEST(uint384, edge_test_eq) {
    uint64_t a[6] = {0, 0, 0, 0, 0, 0};
    uint64_t b[6] = {0, 0, 0, 0, 0, 0};
    uint64_t c[6] = {0, 1, 2, 3, 4, 5};
    uint64_t d[6] = {0, 1, 2, 3, 4, 5};
    uint64_t e[6] = {9, 8, 7, 6, 5, 4};

    
    EXPECT_TRUE(eq(a, a));
    EXPECT_TRUE(eq(b, b));
    EXPECT_TRUE(eq(c, c));
    EXPECT_TRUE(eq(d, d));
    EXPECT_TRUE(eq(e, e));

    EXPECT_TRUE(eq(a, b));
    EXPECT_TRUE(eq(b, a));

    EXPECT_TRUE(eq(c, d));
    EXPECT_TRUE(eq(d, c));


    EXPECT_FALSE(eq(a, c));
    EXPECT_FALSE(eq(c, a));

    EXPECT_FALSE(eq(a, e));
    EXPECT_FALSE(eq(e, a));

    EXPECT_FALSE(eq(c, e));
    EXPECT_FALSE(eq(e, c));

    // one hot bit vectors
    for(uint i = 0; i < 384 - 1; i++){
        uint a_word = i / 64;
        uint a_bit = i % 64;

        uint b_word = (i+1) / 64;
        uint b_bit = (i+1) % 64;

        a[a_word] = 1 << a_bit;
        b[b_word] = 1 << b_bit;

        EXPECT_FALSE(eq(a, b));
        EXPECT_FALSE(eq(b, a));

        a[a_word] = 0;
        b[b_word] = 0;
    }
}

TEST(uint384, rand_test_eq) {
    uint64_t a[6] = {1,2,3,4,5,6};
    uint64_t b[6] = {1,2,3,4,5,6};

    for(int i = 0; i < test_runs; i++) {
        uint rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        b[word] = rand;

        EXPECT_TRUE(eq(a, b));
        EXPECT_TRUE(eq(b, a));
    }
}


TEST(uint384, edge_test_geq){
    uint64_t a[6] = {0, 0, 0, 0, 0, 0};
    uint64_t b[6] = {0, 0, 0, 0, 0, 0};
    uint64_t c[6] = {0, 1, 2, 3, 4, 5};
    uint64_t d[6] = {0, 1, 2, 3, 4, 5};
    uint64_t e[6] = {9, 8, 7, 6, 5, 4};

    
    EXPECT_TRUE(geq(a, a));
    EXPECT_TRUE(geq(b, b));
    EXPECT_TRUE(geq(c, c));
    EXPECT_TRUE(geq(d, d));
    EXPECT_TRUE(geq(e, e));

    EXPECT_TRUE(geq(a, b));
    EXPECT_TRUE(geq(b, a));

    EXPECT_TRUE(geq(c, d));
    EXPECT_TRUE(geq(d, c));


    EXPECT_FALSE(geq(a, c));
    EXPECT_TRUE(geq(c, a));

    EXPECT_FALSE(geq(a, e));
    EXPECT_TRUE(geq(e, a));

    EXPECT_TRUE(geq(c, e));
    EXPECT_FALSE(geq(e, c));

    // one hot bit vectors
    for(uint64_t s = 1; s < 383; s++){
        for(uint64_t i = 0; i < 383 - s; i++){
            uint64_t a_word = i / 64lu;
            uint64_t a_bit = i % 64;

            uint64_t b_word = (i+s) / 64lu;
            uint64_t b_bit = (i+s) % 64lu;

            a[a_word] = 1lu << a_bit;
            b[b_word] = 1lu << b_bit;

            EXPECT_FALSE(geq(a, b));
            EXPECT_TRUE(geq(b, a));

            a[a_word] = 0;
            b[b_word] = 0;
        }
    }
}

TEST(uint384, rand_test_geq){
    uint64_t a[6] = {1,2,3,4,5,6};
    uint64_t b[6] = {1,2,3,4,5,6};

    for(int i = 0; i < test_runs; i++) {
        uint rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        b[word] = rand;

        EXPECT_TRUE(geq(a, b));
        EXPECT_TRUE(geq(b, a));

        if(b[word] > 0) {
            b[word] = rand - 1;
        }
        else{
            a[word] = rand + 1;
        }

        EXPECT_TRUE(geq(a, b));
        EXPECT_FALSE(geq(b, a));

        a[word] = rand;
        b[word] = rand;
    }
}


TEST(uint384, test_leading_zeros) {
    uint64_t a[6] = {0,0,0,0,0,0};
    EXPECT_EQ(384, leading_zeros(a));

    for(int i = 0; i < test_runs; i++) {
        uint rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        EXPECT_EQ(leading_zeros(a), __builtin_clzll(rand) + 64 * (5-word));
        a[word] = 0;
    }
}

TEST(uint384, test_copy) {
    uint64_t a[6] = {0,0,0,0,0,0};
    uint64_t b[6] = {0,0,0,0,0,0};
    copy(b, a);
    EXPECT_TRUE(eq(a, b));

    for(int i = 0; i < test_runs; i++) {
        uint rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        copy(b, a);
        EXPECT_TRUE(eq(a, b));
        a[word] = 0;
    }
}

TEST(uint384, test_rshift) {
    uint64_t a[6] = {0,0,0,0,0,0};
    uint64_t b[6] = {0,0,0,0,0,0};
    rshift(b, a);
    EXPECT_TRUE(eq(a, b));

    for(int i = 0; i < test_runs; i++) {
        uint64_t rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        rshift(b, a);

        a[word] = rand >> 1;
        if(word > 0) a[word - 1] = (rand & 1) << 63;

        EXPECT_TRUE(eq(a, b));
        a[word] = 0;
        if(word > 0) a[word - 1] = 0;
    }

    memset(a, 0, sizeof(a));
    for(int i = 0; i < test_runs; i++) {
        uint64_t rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        rshift(a, a);
        EXPECT_EQ(a[word], rand >> 1);
        if(word > 0) EXPECT_EQ(a[word - 1], rand << 63);
        memset(a, 0, sizeof(a));
    }
}

TEST(uint384, test_lshift){
    uint64_t a[6] = {0,0,0,0,0,0};
    uint64_t b[6] = {0,0,0,0,0,0};
    lshift(b, a);
    EXPECT_TRUE(eq(a, b));

    for(int i = 0; i < test_runs; i++) {
        uint64_t rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        lshift(b, a);

        a[word] = rand << 1;
        if(word < 5) a[word + 1] = rand >> 63;

        EXPECT_TRUE(eq(a, b));
        a[word] = 0;
        if(word < 5) a[word + 1] = 0;
    }

    memset(a, 0, sizeof(a));
    for(int i = 0; i < test_runs; i++) {
        uint64_t rand = random64();
        uint word = rand % 6;
        a[word] = rand;
        lshift(a, a);
        EXPECT_EQ(a[word], rand << 1);
        if(word < 5) EXPECT_EQ(a[word + 1], rand >> 63);
        memset(a, 0, sizeof(a));
    }
}

TEST(uint384, test_add){
    uint64_t a[6] = {0,0,0,0,0,0};
    uint64_t b[6] = {1,0,0,0,0,0};
    uint64_t temp[6];

    uint64_t f500[6] = {
        0x1e2278b212c93d2d,
        0x0773c33170414e4e,
        0x14b7801fd988dea3,
        0xe46ffa400471515e,
        0x5f5f0dad9359c2b1,
        0x0000000003e3fe61,
    };

    uint64_t f501[6] = {
        0x4371701c8ed8f5c2,
        0x64dd4d98275e4dc5,
        0xdce4e804a384e8dc,
        0x40ca94e9f3d1c3a3,
        0xb4154cf918d24bfd,
        0x00000000064b8d36,
    };

    uint64_t test[6] = {0, 0, 0, 0x123456789abcdef0, 0x123456789abcdef0, 0x123456789abcdef0};

    // Out of place addition
    for(int i = 0; i < 250; i++) { 
        add(temp, a, b);
        copy(a, temp);

        add(temp, b, a);
        copy(b, temp);
    }

    EXPECT_TRUE(eq(a, f500));
    EXPECT_TRUE(eq(b, f501));


    // Left inplace addition
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    b[0] = 1;

    for(int i = 0; i < 250; i++) { 
        add(a, a, b);
        add(b, b, a);
    }

    EXPECT_TRUE(eq(a, f500));
    EXPECT_TRUE(eq(b, f501));


    // Right inplace addition
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    b[0] = 1;
    for(int i = 0; i < 250; i++) { 
        add(a, b, a);
        add(b, a, b);
    }

    EXPECT_TRUE(eq(a, f500));
    EXPECT_TRUE(eq(b, f501));


    // Left-Right inplace addition
    memset(a, 0, sizeof(a));
    a[0] = 0x123456789abcdef0;
    a[1] = 0x123456789abcdef0;
    a[2] = 0x123456789abcdef0;

    for(int i = 0; i < 192; i++) { 
        add(a, a, a);
    }

    EXPECT_TRUE(eq(a, test));
}

TEST(uint384, test_sub){

    uint64_t f501[6] = {
        0x4371701c8ed8f5c2,
        0x64dd4d98275e4dc5,
        0xdce4e804a384e8dc,
        0x40ca94e9f3d1c3a3,
        0xb4154cf918d24bfd,
        0x00000000064b8d36,
    };

    uint64_t f500[6] = {
        0x1e2278b212c93d2d,
        0x0773c33170414e4e,
        0x14b7801fd988dea3,
        0xe46ffa400471515e,
        0x5f5f0dad9359c2b1,
        0x0000000003e3fe61,
    };

    uint64_t f1[6] = {1,0,0,0,0,0};

    uint64_t f0[6] = {0,0,0,0,0,0};


    uint64_t a[6], b[6];
    copy(a, f501);
    copy(b, f500);
    uint64_t temp[6];


    // Out of place subtraction
    for(int i = 0; i < 250; i++) { 
        sub(temp, a, b);
        copy(a, temp);

        sub(temp, b, a);
        copy(b, temp);
    }

    EXPECT_TRUE(eq(a, f1));
    EXPECT_TRUE(eq(b, f0));


    // Left inplace addition
    copy(a, f501);
    copy(b, f500);

    for(int i = 0; i < 250; i++) { 
        sub(a, a, b);
        sub(b, b, a);
    }

    EXPECT_TRUE(eq(a, f1));
    EXPECT_TRUE(eq(b, f0));


    // right inplace addition
    copy(a, f501);
    copy(b, f500);

    for(int i = 0; i < 500; i++) { 
        copy(temp,b);
        sub(b, a, b);
        copy(a, temp);
    }

    EXPECT_TRUE(eq(a, f1));
    EXPECT_TRUE(eq(b, f0));

    // left-right inplace addition
    copy(a, f501);
    copy(b, f500);
    sub(a, a, a);
    sub(b, b, b);
    EXPECT_TRUE(eq(a, f0));
    EXPECT_TRUE(eq(b, f0));
}