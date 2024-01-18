//
//  SegmenAnything.swift
//  SAM
//
//  Created by Alessandro Toschi on 18/01/24.
//

import Foundation
import Metal
import CoreML

public class SegmentAnything {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue
  private let imageProcessor: ImageProcessor
  
  private var width: Int!
  private var height: Int!
  
  private var imageEmbeddings: MLMultiArray!
  private var imageEncoder: ImageEncoder!
  
  private var promptEncoder: PromptEncoder
  private var imagePointEmbeddings: MLMultiArray!
  private var denseEmbeddings: MLMultiArray!
  
  private var maskDecoder: MaskDecoder!
  
  public init(device: MTLDevice) {
    self.device = device
    self.commandQueue = device.makeCommandQueue()!
    self.imageProcessor = ImageProcessor(
      device: device,
      mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
      std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)
    )
    self.promptEncoder = PromptEncoder()
  }
  
  public func load() {
    self.imageProcessor.load()
    
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndGPU
    self.imageEncoder = try! ImageEncoder(configuration: modelConfiguration)
    
    self.imagePointEmbeddings = self.promptEncoder.imagePointEmbeddings()
    self.denseEmbeddings = self.promptEncoder.denseEmbeddings()
    
    modelConfiguration.computeUnits = .all
    self.maskDecoder = try! MaskDecoder(configuration: modelConfiguration)
  }
  
  public func preprocess(image: MTLTexture) {
    self.width = image.width
    self.height = image.height
    
    let resizedImage = self.imageProcessor.preprocess(
      image: image,
      commandQueue: self.commandQueue
    )
    
    let imageEncoderInput = ImageEncoderInput(input: resizedImage)
    let imageEncoderOutput = try! self.imageEncoder.prediction(input: imageEncoderInput)
    
    self.imageEmbeddings = imageEncoderOutput.output
  }
  
  public func predictMask(points: [Point]) {
    let pointEmbeddings = self.promptEncoder.pointEmbeddings(
      points: points,
      width: self.width,
      height: self.height
    )
    
    let maskDecoderInput = MaskDecoderInput(
      imageEmbeddings: self.imageEmbeddings,
      imagePointEmbeddings: self.imagePointEmbeddings,
      sparsePromptEmbeddings: pointEmbeddings,
      densePromptEmbeddings: self.denseEmbeddings
    )
    let maskDecoderOutput = try! self.maskDecoder.prediction(input: maskDecoderInput)
    
    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = self.commandQueue
    captureDescriptor.destination = .developerTools
    try! MTLCaptureManager.shared().startCapture(with: captureDescriptor)
    
    let masks = self.imageProcessor.postprocess(
      masks: maskDecoderOutput.masks,
      commandQueue: self.commandQueue
    )
    
    MTLCaptureManager.shared().stopCapture()

    print(maskDecoderOutput.iou_predictions[0])
    print(maskDecoderOutput.iou_predictions[1])
    print(maskDecoderOutput.iou_predictions[2])
    print(maskDecoderOutput.iou_predictions[3])
  }
}
