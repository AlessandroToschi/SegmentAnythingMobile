//
//  mask.metal
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 24/01/24.
//

#include <metal_stdlib>
using namespace metal;


kernel void mask_kernel(texture2d<float, access::read> image [[ texture(0) ]],
                        texture2d<float, access::read> mask [[ texture(1) ]],
                        texture2d<float, access::read_write> output [[ texture(2) ]],
                        constant bool& addictive [[ buffer(0)]],
                        uint2 xy [[ thread_position_in_grid ]]) {
  if (xy.x >= image.get_width() || xy.y >= image.get_height()) {
    return;
  }
  
  const float mask_value = mask.read(xy).x;
  float4 output_color = (addictive == (mask_value > 0.0f)) ? image.read(xy) : float4(0.0);
  output_color.a *= mask_value;
  
  output.write(output_color, xy);
}
