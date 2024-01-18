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

public class ImageProcessor {
  private let device: MTLDevice
  private let mean: SIMD3<Float>
  private let std: SIMD3<Float>
  private let inputSize: Int
  private let outputSize: Int
  
  private var originalWidth: Int!
  private var originalHeight: Int!
  
  private var preprocessComputePipelineState: MTLComputePipelineState!
  private var inputBuffer: MTLBuffer!
  
  private var postprocessComputePipelineState: MTLComputePipelineState!
  
  public init(
    device: MTLDevice,
    mean: SIMD3<Float>,
    std: SIMD3<Float>
  ) {
    self.device = device
    self.mean = mean
    self.std = std
    self.inputSize = 1024
    self.outputSize = 256
    
    self.preprocessComputePipelineState = nil
  }
  
  public func load() {
    let library = try! self.device.makeDefaultLibrary(bundle: Bundle(for: Self.self))
    
    let preprocessKernelFunction = library.makeFunction(name: "preprocessing_kernel")!
    let postprocessKernelFunction = library.makeFunction(name: "postprocessing_kernel")!
    
    self.preprocessComputePipelineState = try! self.device.makeComputePipelineState(function: preprocessKernelFunction)
    self.postprocessComputePipelineState = try! self.device.makeComputePipelineState(function: postprocessKernelFunction)
  }
  
  public func preprocess(
    image: MTLTexture,
    commandQueue: MTLCommandQueue
  ) -> MLMultiArray {
    self.originalWidth = image.width
    self.originalHeight = image.height
    
    let scale = Double(self.inputSize) / Double(max(self.originalWidth, self.originalHeight))
    let width = Int(Double(self.originalWidth) * scale + 0.5)
    let height = Int(Double(self.originalHeight) * scale + 0.5)
    let offsetX = (self.inputSize - width) / 2
    let offsetY = (self.inputSize - height) / 2
    
    let channels = 3
    let bytesPerChannel = MemoryLayout<Float>.stride * self.inputSize * self.inputSize
    let bytesCount = channels * bytesPerChannel
    
    if self.inputBuffer == nil || self.inputBuffer.length != bytesCount {
      self.inputBuffer = self.device.makeBuffer(
        length: bytesCount,
        options: .storageModeShared
      )!
    }
    
    var preprocessingInput = PreprocessingInput(
      mean: self.mean,
      std: self.std,
      size: SIMD2<UInt32>(UInt32(self.inputSize), UInt32(self.inputSize)),
      offset: SIMD2<UInt32>(UInt32(offsetX), UInt32(offsetY))
    )
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(self.preprocessComputePipelineState)
    computeCommandEncoder.setTexture(image, index: 0)
    computeCommandEncoder.setBytes(
      &preprocessingInput,
      length: MemoryLayout<PreprocessingInput>.stride,
      index: 0
    )
    computeCommandEncoder.setBuffer(
      self.inputBuffer,
      offset: 0,
      attributeStride: MemoryLayout<Float>.stride,
      index: 1
    )
    computeCommandEncoder.setBuffer(
      self.inputBuffer,
      offset: bytesPerChannel,
      attributeStride: MemoryLayout<Float>.stride,
      index: 2
    )
    computeCommandEncoder.setBuffer(
      self.inputBuffer,
      offset: 2 * bytesPerChannel,
      attributeStride: MemoryLayout<Float>.stride,
      index: 3
    )
    
    let threadgroupSize = MTLSize(
      width: self.preprocessComputePipelineState.threadExecutionWidth,
      height: self.preprocessComputePipelineState.maxTotalThreadsPerThreadgroup / self.preprocessComputePipelineState.threadExecutionWidth,
      depth: 1
    )
    
    if self.device.supportsFamily(.common3) {
      computeCommandEncoder.dispatchThreads(
        MTLSize(
          width: self.inputSize,
          height: self.inputSize,
          depth: 1
        ),
        threadsPerThreadgroup: threadgroupSize
      )
    } else {
      let gridSize = MTLSize(
        width: (self.inputSize + threadgroupSize.width - 1) / threadgroupSize.width,
        height: (self.inputSize + threadgroupSize.height - 1) / threadgroupSize.height,
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
    
    return try! MLMultiArray(
      dataPointer: self.inputBuffer.contents(),
      shape: [
        1,
        channels as NSNumber,
        self.inputSize as NSNumber,
        self.inputSize as NSNumber
      ],
      dataType: .float32,
      strides: [
        (channels * inputSize * inputSize) as NSNumber,
        (inputSize * inputSize) as NSNumber,
        inputSize as NSNumber,
        1
      ]
    )
  }
  
  public func postprocess(masks: MLMultiArray, commandQueue: MTLCommandQueue) -> [MTLTexture] {
    let scale = Float(self.outputSize) / Float(max(self.originalWidth, self.originalHeight))
    let scaledWidth = (Float(self.originalWidth) * scale).rounded()
    let scaledHeight = (Float(originalHeight) * scale).rounded()
    var scaledSize = SIMD2<Float>(scaledWidth, scaledHeight)
    
    var outputMasks = [MTLTexture]()
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    
    let threadgroupSize = MTLSize(
      width: self.preprocessComputePipelineState.threadExecutionWidth,
      height: self.preprocessComputePipelineState.maxTotalThreadsPerThreadgroup / self.preprocessComputePipelineState.threadExecutionWidth,
      depth: 1
    )
    
    masks.withUnsafeMutableBytes {
      pointer,
      strides in
      
      let maskPointer = pointer.bindMemory(to: Float.self).baseAddress!
      let maskStride = strides[1]
      
      let outputMaskTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r32Float,
        width: self.originalWidth,
        height: self.originalHeight,
        mipmapped: false
      )
      outputMaskTextureDescriptor.usage = [.shaderRead, .shaderWrite]
      outputMaskTextureDescriptor.storageMode = .shared
      
      let maskTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r32Float,
        width: masks.shape[3].intValue,
        height: masks.shape[2].intValue,
        mipmapped: false
      )
      maskTextureDescriptor.storageMode = .shared
      
      for maskIndex in 0 ..< masks.shape[1].intValue {
        let maskBuffer = self.device.makeBuffer(
          bytesNoCopy: maskPointer + maskIndex * maskStride,
          length: maskStride * MemoryLayout<Float>.stride
        )!
        let maskTexture = maskBuffer.makeTexture(
          descriptor: maskTextureDescriptor,
          offset: 0,
          bytesPerRow: strides[2] * MemoryLayout<Float>.stride
        )!
        let outputMaskTexture = self.device.makeTexture(descriptor: outputMaskTextureDescriptor)!
        
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeCommandEncoder.setComputePipelineState(self.postprocessComputePipelineState)
        computeCommandEncoder.setTexture(maskTexture, index: 0)
        computeCommandEncoder.setTexture(outputMaskTexture, index: 1)
        computeCommandEncoder.setBytes(&scaledSize, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

        if self.device.supportsFamily(.common3) {
          computeCommandEncoder.dispatchThreads(
            MTLSize(
              width: self.originalWidth,
              height: self.originalHeight,
              depth: 1
            ),
            threadsPerThreadgroup: threadgroupSize
          )
        } else {
          let gridSize = MTLSize(
            width: (self.originalWidth + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (self.originalHeight + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
          )
          computeCommandEncoder.dispatchThreadgroups(
            gridSize,
            threadsPerThreadgroup: threadgroupSize
          )
        }
        
        computeCommandEncoder.endEncoding()
      }
    }
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return outputMasks
  }
}
