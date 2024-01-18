//
//  Preprocessing.h
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 15/01/24.
//

#ifndef Preprocessing_h
#define Preprocessing_h

#include <simd/simd.h>

struct PreprocessingInput {
  simd_float3 mean;
  simd_float3 std;
  simd_uint2 size;
  simd_uint2 offset;
};

#endif /* Preprocessing_h */
