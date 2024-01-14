//
//  ScalarFilter.swift
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 11/01/24.
//

import Foundation
import MetalPerformanceShaders

class NormalizeFilter: MPSUnaryImageKernel {
  var mean: SIMD3<Float>
  var std: SIMD3<Float>
  
  private var computePipelineState: MTLComputePipelineState!
  
  init(
    mean: SIMD3<Float>,
    std: SIMD3<Float>,
    device: MTLDevice
  ) {
    self.mean = mean
    self.std = std
    super.init(device: device)
  }
  
  required init?(coder aDecoder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }
  
  override func encode(
    commandBuffer: MTLCommandBuffer,
    inPlaceTexture texture: UnsafeMutablePointer<MTLTexture>,
    fallbackCopyAllocator copyAllocator: MPSCopyAllocator? = nil
  ) -> Bool {
    guard let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
    else { return false }
    
    if self.computePipelineState == nil {
      guard let library = try? self.device.makeDefaultLibrary(bundle: Bundle(for: NormalizeFilter.self)),
            let function = library.makeFunction(name: "normalize_kernel"),
            let computePipelineState = try? self.device.makeComputePipelineState(function: function)
      else { return false }
      self.computePipelineState = computePipelineState
    }
    
    computeCommandEncoder.setComputePipelineState(self.computePipelineState)
    computeCommandEncoder.setTexture(texture.pointee, index: 0)
    computeCommandEncoder.setBytes(
      &self.mean,
      length: MemoryLayout<SIMD3<Float>>.size,
      index: 0
    )
    computeCommandEncoder.setBytes(
      &self.std,
      length: MemoryLayout<SIMD3<Float>>.size,
      index: 1
    )
    computeCommandEncoder.dispatchThreads(
      MTLSize(
        width: texture.pointee.width,
        height: texture.pointee.height,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(
        width: self.computePipelineState.threadExecutionWidth,
        height: self.computePipelineState.maxTotalThreadsPerThreadgroup / self.computePipelineState.threadExecutionWidth,
        depth: 1
      )
    )
    computeCommandEncoder.endEncoding()
    
    return true
  }
}
