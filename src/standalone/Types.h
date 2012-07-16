#ifndef TYPES_H_
#define TYPES_H_

#include <complex>
#include <cmath>
#include <cstdio>
#include <boost/cstdint.hpp>

namespace TYPES {

// Convenience shortcuts.
// TODO these don't seem to be used anywhere!?
typedef unsigned char        uchar;
typedef unsigned short       ushort;
typedef unsigned int         uint;
typedef unsigned long        ulong;
typedef long long            longlong;
typedef unsigned long long   ulonglong;
typedef long double          ldouble;

// Fixed data sizes.
// TODO: needed? these are defined in <boost/cstdint.hpp>
// too (see ones used in the UDPPacket). Also perhaps the Qt versions qint8 etc.
// might be more portable... ?
typedef signed char         int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;

/// Container for 16bit signed complex numbers.
typedef std::complex<int8>   i8complex;

/// Container for 32bit signed complex numbers.
typedef std::complex<int16>  i16complex;


/**
 * @class i4complex
 *
 * @ingroup pelican_lofar
 *
 * @brief
 * Container class for 8bit signed packed complex numbers.
 *
 * @details
 * This class stores complex numbers in a packed 8 bit format where the real
 * and imaginary parts are each signed 4bit integers.
 *
 * In this format real and imaginary numbers are represented by 16 discrete
 * values.
 *
 * @ref
 * Not sure this is right but the class seems to come from here:
 * http://www.lofar.org/software/docxxhtml/classLOFAR_1_1TYPES_1_1i4complex.html
 */
class i4complex
{
    public:
		/// Default Constructor.
        i4complex() {}

        /// Constructs a complex number from the specified real and imaginary
        /// parts. (TODO: is rint() portable?)
        i4complex(double real, double imag) {
            value = ((int) rint(real - .5) & 0xF)  | (((int) rint(imag - .5) & 0xF) << 4);
        }

        /// Returns the real part.
        double real() const {
            return ((signed char) (value << 4) >> 4) + .5; // extend sign
        }

        /// Returns the imaginary part.
        double imag() const {
            return (value >> 4) + .5;
        }

        /// Returns the complex conjugate.
        i4complex conj() const {
            return i4complex(value ^ 0xF0);
        }

    private:
        i4complex(unsigned char value)
        : value(value) {}

        // Do not use bitfields for the real and imaginary parts, since the
        // allocation order is not portable between different compilers
        signed char value;
};

// Functions operating on int4 complex numbers.
// <group>
inline double real(TYPES::i4complex v)
{ return v.real(); }
inline double imag(TYPES::i4complex v)
{ return v.imag(); }
inline TYPES::i4complex conj(TYPES::i4complex x)
{ return x.conj(); }
// </group>

// Functions to make complex numbers.
// <group>
inline TYPES::i4complex makei4complex(double real, double imag)
{ return TYPES::i4complex(real, imag); }
inline TYPES::i8complex makei8complex (TYPES::int8 re, TYPES::int8 im)
{ return TYPES::i8complex(re, im); }
inline TYPES::i16complex makei16complex (TYPES::int16 re, TYPES::int16 im)
{ return TYPES::i16complex(re, im); }
// </group>

// Functions operating on int8 complex numbers.
// <group>
inline TYPES::int8 real (TYPES::i8complex x)
{ return x.real(); }
inline TYPES::int8 imag (TYPES::i8complex x)
{ return x.imag(); }
inline TYPES::i8complex conj (TYPES::i8complex x)
{ return conj(x); }
// </group>

// Functions operating on int16 complex numbers.
// <group>
inline TYPES::int16 real (TYPES::i16complex x)
{ return x.real(); }
inline TYPES::int16 imag (TYPES::i16complex x)
{ return x.imag(); }
inline TYPES::i16complex conj (TYPES::i16complex x)
{ return conj(x); }
// </group>

} // namespace TYPES

#endif /* TYPES_H_ */
