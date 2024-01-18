//
//  postprocessing.metal
//  SAM
//
//  Created by Alessandro Toschi on 18/01/24.
//

#include <metal_stdlib>
using namespace metal;

constexpr sampler linear_sampler = sampler(filter::linear);

kernel void postprocessing_kernel(texture2d<float, access::sample> mask [[ texture(0) ]],
                                  texture2d<float, access::read_write> output [[ texture (1) ]],
                                  constant float2& mask_size [[ buffer(0) ]],
                                  uint2 xy [[ thread_position_in_grid] ]) {
  
  if (xy.x >= output.get_width() || xy.y >= output.get_height()) {
    return;
  }
  
  const float2 uv = float2(xy) / float2(output.get_width(), output.get_height());
  output.write(step(mask.sample(linear_sampler, uv), 0.0), xy);
}
