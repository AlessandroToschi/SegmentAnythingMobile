//
//  preprocessing.metal
//  SAM
//
//  Created by Alessandro Toschi on 15/01/24.
//

#include <metal_stdlib>
#include "Preprocessing-Bridging-Header.h"

using namespace metal;

constexpr sampler preprocessing_sampler = sampler(filter::linear);

kernel void preprocessing_kernel(texture2d<float, access::sample> texture [[ texture(0) ]],
                                 constant PreprocessingInput& input [[ buffer(0) ]],
                                 device float* rBuffer [[ buffer(1) ]],
                                 device float* gBuffer [[ buffer(2) ]],
                                 device float* bBuffer [[ buffer(3) ]],
                                 uint2 xy [[ thread_position_in_grid ]]
                                 ) {
  if (xy.x >= input.size.x || xy.y >= input.size.y) {
    return;
  }
  
  const float2 uv = float2(xy) / float2(input.size);
  float4 color = texture.sample(preprocessing_sampler, uv);
  color.rgb = (color.rgb - input.mean) / input.std;
  
  const int index = (xy.y + input.offset.y) * input.size.x + xy.x + input.offset.x;
  
  rBuffer[index] = color.r;
  gBuffer[index] = color.g;
  bBuffer[index] = color.b;
}
