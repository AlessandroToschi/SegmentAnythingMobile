//
//  ImageProcessor-Bridging-Header.h
//  SAM
//
//  Created by Alessandro Toschi on 22/01/24.
//

#ifndef ImageProcessor_Bridging_Header_h
#define ImageProcessor_Bridging_Header_h

#import <simd/simd.h>

struct PreprocessingInput {
  simd_float3 mean;
  simd_float3 std;
  simd_uint2 size;
  simd_uint2 padding;
};

struct PostprocessingInput {
  simd_float2 scaleSizeFactor;
};

#endif /* ImageProcessor_Bridging_Header_h */
