//
//  filters.metal
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 11/01/24.
//

#include <metal_stdlib>
using namespace metal;

constexpr sampler normalize_resample_sampler = sampler(filter::linear);

kernel void normalize_resampling_kernel(texture2d<float, access::sample> input_texture [[ texture(0) ]],
                                      texture2d<float, access::read_write> output_texture [[ texture(1) ]],
                                      constant float3& mean [[ buffer(0) ]],
                                      constant float3& std [[ buffer(1) ]],
                                      uint2 xy [[thread_position_in_grid]]
                                      ) {
  if (xy.x >= output_texture.get_width() || xy.y >= output_texture.get_height()) {
    return;
  }
  
  const float2 uv = float2(xy) / float2(output_texture.get_width(), output_texture.get_height());
  float4 color = input_texture.sample(normalize_resample_sampler, uv);
  color.rgb = (color.rgb - mean) / std;
  output_texture.write(color, xy);
}

kernel void normalize_kernel(texture2d<float, access::read_write> texture [[texture(0)]],
                             constant float3& mean [[buffer(0)]],
                             constant float3& std [[buffer(1)]],
                             uint2 index [[thread_position_in_grid]]
                             ) {
  float4 color = texture.read(index);
  color.rgb = (color.rgb - mean) / std;
  texture.write(color, index);
}
