#ifndef REAL_H
#define REAL_H

// A way to make the code independent of the data type used for floating point numbers.

namespace mif {
    #ifdef REAL_FLOAT
        #define Real float
    #else
    #define Real double
    #endif
    
} //mif

#endif // REAL_H