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

class ImageProcessor {
  private let device: MTLDevice
  private let mean: SIMD3<Float>
  private let std: SIMD3<Float>
  private let inputSize: Int
  private let outputSize: Int
  
  private var originalWidth: Int!
  private var originalHeight: Int!
  
  private var resizedWidth: Int!
  private var resizedHeight: Int!
  
  private var preprocessComputePipelineState: MTLComputePipelineState!
  private var inputBuffer: MTLBuffer!
  
  private var postprocessComputePipelineState: MTLComputePipelineState!
  
  init(
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
  
  func load() {
#if os(iOS)
    let libraryName = "image_processor_ios"
#elseif os(macOS)
    let libraryName = "image_processor_macos"
#endif
    
    let libraryUrl = Bundle(for: Self.self).url(
      forResource: libraryName,
      withExtension: "metallib"
    )!
    
    let library = try! self.device.makeLibrary(URL: libraryUrl)
    
    let preprocessKernelFunction = library.makeFunction(name: "preprocessing_kernel")!
    let postprocessKernelFunction = library.makeFunction(name: "postprocessing_kernel")!
    
    self.preprocessComputePipelineState = try! self.device.makeComputePipelineState(function: preprocessKernelFunction)
    self.postprocessComputePipelineState = try! self.device.makeComputePipelineState(function: postprocessKernelFunction)
  }
  
  func preprocess(
    image: MTLTexture,
    commandQueue: MTLCommandQueue
  ) -> MLMultiArray {
    self.originalWidth = image.width
    self.originalHeight = image.height
    
    let scale = Double(self.inputSize) / Double(max(self.originalWidth, self.originalHeight))
    self.resizedWidth = Int(Double(self.originalWidth) * scale + 0.5)
    self.resizedHeight = Int(Double(self.originalHeight) * scale + 0.5)
    let paddingX = self.inputSize - self.resizedWidth
    let paddingY = self.inputSize - self.resizedHeight
    
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
      size: SIMD2<UInt32>(UInt32(self.resizedWidth), UInt32(self.resizedHeight)),
      padding: SIMD2<UInt32>(UInt32(paddingX), UInt32(paddingY))
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
          width: self.resizedWidth,
          height: self.resizedHeight,
          depth: 1
        ),
        threadsPerThreadgroup: threadgroupSize
      )
    } else {
      let gridSize = MTLSize(
        width: (self.resizedWidth + threadgroupSize.width - 1) / threadgroupSize.width,
        height: (self.resizedHeight + threadgroupSize.height - 1) / threadgroupSize.height,
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
  
  func postprocess(masks: MLMultiArray, commandQueue: MTLCommandQueue) -> [MTLTexture] {
    let scale = Float(self.outputSize) / Float(max(self.originalWidth, self.originalHeight))
    let scaledWidth = (Float(self.originalWidth) * scale).rounded()
    let scaledHeight = (Float(self.originalHeight) * scale).rounded()
    let scaleSizeFactor = SIMD2<Float>(
      scaledWidth / Float(self.outputSize),
      scaledHeight / Float(self.outputSize)
    )
    
    var outputMasks = [MTLTexture]()
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    
    let threadgroupSize = MTLSize(
      width: self.preprocessComputePipelineState.threadExecutionWidth,
      height: self.preprocessComputePipelineState.maxTotalThreadsPerThreadgroup / self.preprocessComputePipelineState.threadExecutionWidth,
      depth: 1
    )
    let gridSizeOrThreads: MTLSize
    
    if self.device.supportsFamily(.common3) {
      gridSizeOrThreads = MTLSize(
        width: self.originalWidth,
        height: self.originalHeight,
        depth: 1
      )
    } else {
      gridSizeOrThreads = MTLSize(
        width: (self.originalWidth + threadgroupSize.width - 1) / threadgroupSize.width,
        height: (self.originalHeight + threadgroupSize.height - 1) / threadgroupSize.height,
        depth: 1
      )
    }
    
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
    
    var postprocessingInput = PostprocessingInput(scaleSizeFactor: scaleSizeFactor)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(self.postprocessComputePipelineState)
    computeCommandEncoder.setBytes(
      &postprocessingInput,
      length: MemoryLayout<PostprocessingInput>.stride,
      index: 0
    )
    
    masks.withUnsafeMutableBytes {
      pointer,
      strides in
      
      let maskPointer = pointer.bindMemory(to: Float.self).baseAddress!
      let maskStride = strides[1]
      
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
        
        computeCommandEncoder.setTexture(maskTexture, index: 0)
        computeCommandEncoder.setTexture(outputMaskTexture, index: 1)
        
        if self.device.supportsFamily(.common3) {
          computeCommandEncoder.dispatchThreads(
            gridSizeOrThreads,
            threadsPerThreadgroup: threadgroupSize
          )
        } else {
          computeCommandEncoder.dispatchThreadgroups(
            gridSizeOrThreads,
            threadsPerThreadgroup: threadgroupSize
          )
        }
        
        outputMasks.append(outputMaskTexture)
      }
    }
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    return outputMasks
  }
  
  func mapPoints(points: [Point]) -> PromptEncoderInput {
    let pointsShape = [1, points.count, 2]
    let labelsShape = [1, points.count]
    let scaleWidth = Float(self.resizedWidth) / Float(self.originalWidth)
    let scaleHeight = Float(self.resizedHeight) / Float(self.originalHeight)
    
    let pointsTensor = MLShapedArray<Float>(
      scalars: points.map({ [$0.x * scaleWidth, $0.y * scaleHeight] })
        .reduce(into: [], { $0.append(contentsOf: $1) }),
      shape: pointsShape
    )
    let labelsTensor = MLShapedArray<Float>(
      scalars: points.map({ Float($0.label) }),
      shape: labelsShape
    )
    
    return PromptEncoderInput(
      points: pointsTensor,
      labels: labelsTensor
    )
  }
  
  func loadTensor(tensorName: String, tensorExtension: String = "bin", shape: [Int]) -> MLMultiArray {
    let count = shape.reduce(1, *)
    let strides = shape.reduce([Int](), { return $0 + [($0.last ?? 1) * $1] }).reversed()
    
    let tensorUrl = Bundle(for: Self.self).url(
      forResource: tensorName,
      withExtension: tensorExtension
    )!
    
    let tensorPointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
    
    let tensorData = try! Data(contentsOf: tensorUrl)
    (tensorData as NSData).getBytes(tensorPointer, length: tensorData.count)
    
    return try! MLMultiArray(
      dataPointer: tensorPointer,
      shape: shape as [NSNumber],
      dataType: .float32,
      strides: strides.reversed() as [NSNumber],
      deallocator: { $0.deallocate() }
    )
  }
}
