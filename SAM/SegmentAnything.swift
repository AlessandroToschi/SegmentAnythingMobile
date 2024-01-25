//
//  SegmenAnything.swift
//  SAM
//
//  Created by Alessandro Toschi on 18/01/24.
//

import Foundation
import Metal
import CoreML

public struct Point: Equatable {
  var x: Float
  var y: Float
  var label: Int
  
  public init(
    x: Float,
    y: Float,
    label: Int
  ) {
    self.x = x
    self.y = y
    self.label = label
  }
}

public class SegmentAnything {
  public let device: MTLDevice
  
  private let commandQueue: MTLCommandQueue
  private let imageProcessor: ImageProcessor
  
  private var width: Int!
  private var height: Int!
  
  private var imageEmbeddings: MLMultiArray!
  private var imageEncoder: ImageEncoder!
  
  private var promptEncoder: PromptEncoder!
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
  }
  
  public func load() {
    self.imageProcessor.load()
    
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndGPU
    self.imageEncoder = try! ImageEncoder(configuration: modelConfiguration)
    
    self.imagePointEmbeddings = self.imageProcessor.loadTensor(
      tensorName: "image_embeddings",
      shape: [1, 256, 64, 64]
    )
    
    self.promptEncoder = try! PromptEncoder()
    
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
  
  public func predictMasks(points: [Point]) -> [(MTLTexture, Float)] {
    let pEncoderOutput = try! self.promptEncoder.prediction(
      input: self.imageProcessor.mapPoints(points: points)
    )
    
    let maskDecoderInput = MaskDecoderInput(
      imageEmbeddings: self.imageEmbeddings,
      imagePointEmbeddings: self.imagePointEmbeddings,
      sparsePromptEmbeddings: pEncoderOutput.sparseEmbeddings,
      densePromptEmbeddings: pEncoderOutput.denseEmbeddings
    )
    let maskDecoderOutput = try! self.maskDecoder.prediction(input: maskDecoderInput)
    
    let masks = self.imageProcessor.postprocess(
      masks: maskDecoderOutput.masks,
      commandQueue: self.commandQueue
    )
    
    return zip(
      masks,
      maskDecoderOutput.iou_predictionsShapedArray.scalars
    ).reduce(into: [], { $0.append(($1.0, $1.1))})
  }
}
