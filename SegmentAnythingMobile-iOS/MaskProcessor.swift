//
//  Mask.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 24/01/24.
//

import Foundation
import Metal
import CoreImage
import UIKit

class MaskProcessor {
  enum Mode {
    case addictive
    case subtractive
  }
  
  private let device: MTLDevice
  
  private var commandQueue: MTLCommandQueue!
  private var maskComputePipelineState: MTLComputePipelineState!
  private var context: CIContext!
  
  init(device: MTLDevice) {
    self.device = device
  }
  
  func load() {
    self.commandQueue = self.device.makeCommandQueue()!
    
    let library = self.device.makeDefaultLibrary()!
    let maskKernelFunction = library.makeFunction(name: "mask_kernel")!
    
    self.maskComputePipelineState = try! self.device.makeComputePipelineState(function: maskKernelFunction)
    
    self.context = CIContext(mtlDevice: self.device)
  }
  
  func apply(input: MTLTexture, mask: MTLTexture, mode: Mode) -> MTLTexture {
    assert(input.width == mask.width)
    assert(input.height == mask.height)
    
    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = self.commandQueue
    
    let captureDevice = MTLCaptureManager.shared()
    //try! captureDevice.startCapture(with: captureDescriptor)
    
    let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: input.width,
      height: input.height,
      mipmapped: false
    )
    outputTextureDescriptor.usage = [.shaderRead, .shaderWrite]
    outputTextureDescriptor.storageMode = .shared
    
    let outputTexture = self.device.makeTexture(descriptor: outputTextureDescriptor)!
    
    var addictive = mode == .addictive
    
    let commandBuffer = self.commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(self.maskComputePipelineState)
    computeCommandEncoder.setTexture(input, index: 0)
    computeCommandEncoder.setTexture(mask, index: 1)
    computeCommandEncoder.setTexture(outputTexture, index: 2)
    computeCommandEncoder.setBytes(&addictive, length: MemoryLayout<Bool>.stride, index: 0)
    
    let threadgroupSize = MTLSize(
      width: self.maskComputePipelineState.threadExecutionWidth,
      height: self.maskComputePipelineState.maxTotalThreadsPerThreadgroup / self.maskComputePipelineState.threadExecutionWidth,
      depth: 1
    )
    
    if self.device.supportsFamily(.common3) {
      computeCommandEncoder.dispatchThreads(
        MTLSize(
          width: input.width,
          height: input.height,
          depth: 1
        ),
        threadsPerThreadgroup: threadgroupSize
      )
    } else {
      let gridSize = MTLSize(
        width: (input.width + threadgroupSize.width - 1) / threadgroupSize.width,
        height: (input.height + threadgroupSize.height - 1) / threadgroupSize.height,
        depth: 1
      )
      computeCommandEncoder.dispatchThreadgroups(
        gridSize,
        threadsPerThreadgroup: threadgroupSize
      )
    }
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    return outputTexture
  }
}
