//
//  PromptEncoder.swift
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 10/01/24.
//

import Foundation
import CoreML
import MetalPerformanceShaders

public class SegmentAnythingMobile {
  private let targetSize: Int = 1024
  
  public private(set) var device: MTLDevice
  private var commandQueue: MTLCommandQueue
  
  public init(device: MTLDevice) {
    self.device = device
    self.commandQueue = self.device.makeCommandQueue()!
  }
  
  public func predict(image: MTLTexture) -> MTLTexture {
    #if DEBUG
    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = self.commandQueue
    captureDescriptor.destination = .developerTools
    try! MTLCaptureManager.shared().startCapture(with: captureDescriptor)
    #endif
    
    let commandBuffer = self.commandQueue.makeCommandBuffer()!
    
    let originalWidth = image.width
    let originalHeight = image.height
    
    let scale = Double(self.targetSize) / Double(max(originalWidth, originalHeight))
    let width = Int(Double(originalWidth) * scale + 0.5)
    let height = Int(Double(originalHeight) * scale + 0.5)
    
    let resizedTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba32Float,
      width: width,
      height: height,
      mipmapped: false
    )
    resizedTextureDescriptor.usage = [.shaderRead, .shaderWrite]
    
    let resizedTexture = self.device.makeTexture(descriptor: resizedTextureDescriptor)!
    
    let normalizeResamplingFilter = NormalizeResamplingFilter(
      mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
      std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0),
      device: self.device
    )
    normalizeResamplingFilter.encode(
      commandBuffer: commandBuffer,
      sourceTexture: image,
      destinationTexture: resizedTexture
    )
    /*
    let resizedImageDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: image.pixelFormat,
      width: width,
      height: height,
      mipmapped: false
    )
    resizedImageDescriptor.usage = [.shaderRead, .shaderWrite]
    var resizedTexture = self.device.makeTexture(descriptor: resizedImageDescriptor)!
    
    var resizeTransform = MPSScaleTransform(
      scaleX: scale,
      scaleY: scale,
      translateX: 0.0,
      translateY: 0.0
    )
    
    let resizeFilter = MPSImageBilinearScale(device: self.device)
    withUnsafePointer(to: &resizeTransform) {
      resizeFilter.scaleTransform = $0
      resizeFilter.encode(
        commandBuffer: commandBuffer,
        sourceTexture: image,
        destinationTexture: resizedTexture
      )
    }
    
    let fmaFilter = NormalizeFilter(
      mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
      std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0),
      device: self.device
    )
    fmaFilter.encode(
      commandBuffer: commandBuffer,
      inPlaceTexture: &resizedTexture,
      fallbackCopyAllocator: nil
    )
    */
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    #if DEBUGs
    MTLCaptureManager.shared().stopCapture()
    #endif

    return resizedTexture
  }
}
