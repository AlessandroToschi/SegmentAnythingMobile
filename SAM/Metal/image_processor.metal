//
//  image_processor.metal
//  SAM
//
//  Created by Alessandro Toschi on 22/01/24.
//

#include <metal_stdlib>
#include "ImageProcessor-Bridging-Header.h"

using namespace metal;

constexpr sampler linear_sampler = sampler(filter::linear);

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
  float4 color = texture.sample(linear_sampler, uv);
  color.rgb = (color.rgb - input.mean) / input.std;
  
  const int index = xy.y * (input.size.x + input.padding.x) + xy.x + input.padding.y;
  
  rBuffer[index] = color.r;
  gBuffer[index] = color.g;
  bBuffer[index] = color.b;
}

kernel void postprocessing_kernel(texture2d<float, access::sample> mask [[ texture(0) ]],
                                  texture2d<float, access::read_write> output [[ texture (1) ]],
                                  constant PostprocessingInput& input [[ buffer(0) ]],
                                  uint2 xy [[ thread_position_in_grid] ]) {
  
  if (xy.x >= output.get_width() || xy.y >= output.get_height()) {
    return;
  }
  
  const float2 uv = float2(xy) / float2(output.get_width(), output.get_height());
  const float2 mask_uv = uv * input.scaleSizeFactor;
  const float4 mask_value = 1.0f - step(mask.sample(linear_sampler, mask_uv), 0.0f);
  
  output.write(mask_value, xy);
}
