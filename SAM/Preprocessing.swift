//
//  Preprocessing.swift
//  SAM
//
//  Created by Alessandro Toschi on 15/01/24.
//

import Foundation
import Metal
import MetalPerformanceShaders
import CoreML

public class Preprocessing {
  private let device: MTLDevice
  private let mean: SIMD3<Float>
  private let std: SIMD3<Float>
  private let size: Int
  
  private var computePipelineState: MTLComputePipelineState!
  private var imageEncoder: ImageEncoder!
  
  public init(
    device: MTLDevice,
    mean: SIMD3<Float>,
    std: SIMD3<Float>
  ) {
    self.device = device
    self.mean = mean
    self.std = std
    self.size = 1024
    
    self.computePipelineState = nil
  }
  
  public func load() {
    let library = try! self.device.makeDefaultLibrary(bundle: Bundle(for: Self.self))
    let function = library.makeFunction(name: "preprocessing_kernel")!
    self.computePipelineState = try! self.device.makeComputePipelineState(function: function)
    
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndGPU
    self.imageEncoder = try! ImageEncoder(configuration: modelConfiguration)
  }
  
  public func preprocess(
    image: MTLTexture,
    commandBuffer: MTLCommandBuffer
  ) -> MLMultiArray {
    let originalWidth = image.width
    let originalHeight = image.height
    
    let scale = Double(self.size) / Double(max(originalWidth, originalHeight))
    let width = Int(Double(originalWidth) * scale + 0.5)
    let height = Int(Double(originalHeight) * scale + 0.5)
    let offsetX = (self.size - width) / 2
    let offsetY = (self.size - height) / 2
    
    let channels = 3
    let bytesPerChannel = MemoryLayout<Float>.stride * self.size * self.size
    let bytesCount = channels * bytesPerChannel
    
    let buffer = self.device.makeBuffer(
      length: bytesCount,
      options: .storageModeShared
    )!
    
    var preprocessingInput = PreprocessingInput(
      mean: self.mean,
      std: self.std,
      size: SIMD2<UInt32>(UInt32(self.size), UInt32(self.size)),
      offset: SIMD2<UInt32>(UInt32(offsetX), UInt32(offsetY))
    )
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(self.computePipelineState)
    computeCommandEncoder.setTexture(image, index: 0)
    computeCommandEncoder.setBytes(
      &preprocessingInput,
      length: MemoryLayout<PreprocessingInput>.stride,
      index: 0
    )
    computeCommandEncoder.setBuffer(
      buffer,
      offset: 0,
      attributeStride: MemoryLayout<Float>.stride,
      index: 1
    )
    computeCommandEncoder.setBuffer(
      buffer,
      offset: bytesPerChannel,
      attributeStride: MemoryLayout<Float>.stride,
      index: 2
    )
    computeCommandEncoder.setBuffer(
      buffer,
      offset: 2 * bytesPerChannel,
      attributeStride: MemoryLayout<Float>.stride,
      index: 3
    )
    
    let threadgroupSize = MTLSize(
      width: self.computePipelineState.threadExecutionWidth,
      height: self.computePipelineState.maxTotalThreadsPerThreadgroup / self.computePipelineState.threadExecutionWidth,
      depth: 1
    )
    
    if self.device.supportsFamily(.common3) {
      computeCommandEncoder.dispatchThreads(
        MTLSize(
          width: self.size,
          height: self.size,
          depth: 1
        ),
        threadsPerThreadgroup: threadgroupSize
      )
    } else {
      let gridSize = MTLSize(
        width: (self.size + threadgroupSize.width - 1) / threadgroupSize.width,
        height: (self.size + threadgroupSize.height - 1) / threadgroupSize.height,
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
    
    let imageEncoderInput = ImageEncoderInput(
      input: try! MLMultiArray(
        dataPointer: buffer.contents(),
        shape: [
          1,
          channels as NSNumber,
          self.size as NSNumber,
          self.size as NSNumber
        ],
        dataType: .float32,
        strides: [
          (channels * size * size) as NSNumber,
          (size * size) as NSNumber,
          size as NSNumber,
          1
        ],
        deallocator: { $0.deallocate() }
      )
    )
    
    let imageEncoderOutput = try! imageEncoder.prediction(input: imageEncoderInput)
    return imageEncoderOutput.output
  }
}
