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
  
  private var promptEncoder: PromptEncoderS
  private var pEncoder: PromptEncoder!
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
    self.promptEncoder = PromptEncoderS()
  }
  
  public func load() {
    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = self.commandQueue
    captureDescriptor.destination = .developerTools
    try! MTLCaptureManager.shared().startCapture(with: captureDescriptor)
    
    self.imageProcessor.load()
    
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndGPU
    self.imageEncoder = try! ImageEncoder(configuration: modelConfiguration)
    
    self.imagePointEmbeddings = self.promptEncoder.imagePointEmbeddings()
    //self.denseEmbeddings = self.promptEncoder.denseEmbeddings()
    
    self.pEncoder = try! PromptEncoder()
    
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
    
    let resizedImageUrl = URL.documentsDirectory.appending(path: "image.bin")
    resizedImage.withUnsafeBytes { pointer in
      try! Data(pointer).write(to: resizedImageUrl)
    }
    
    let imageEncoderInput = ImageEncoderInput(input: resizedImage)
    let imageEncoderOutput = try! self.imageEncoder.prediction(input: imageEncoderInput)
    
    self.imageEmbeddings = imageEncoderOutput.output
  }
  
  public func predictMask(points: [Point], outputDirectoryUrl: URL) {
    let pEncoderOutput = try! self.pEncoder.prediction(
      input: points.promptEncoderInput()
    )
    
    /*
     let densePromptEmbeddingsUrl = outputDirectoryUrl.appending(path: "dense_prompt_embeddings.bin")
     self.denseEmbeddings.withUnsafeBytes { pointer in
     try! Data(pointer).write(to: densePromptEmbeddingsUrl)
     }
     
     let sparsePromptEmbeddingsUrl = outputDirectoryUrl.appending(path: "sparse_prompt_embeddings.bin")
     pointEmbeddings.withUnsafeBytes { pointer in
     try! Data(pointer).write(to: sparsePromptEmbeddingsUrl)
     }
     
     let imagePointEmbeddingsUrl = outputDirectoryUrl.appending(path: "image_pe.bin")
     self.imagePointEmbeddings.withUnsafeBytes { pointer in
     try! Data(pointer).write(to: imagePointEmbeddingsUrl)
     }
     
     let imageEmbeddingsUrl = outputDirectoryUrl.appending(path: "image_embeddings.bin")
     self.imageEmbeddings.withUnsafeBytes { pointer in
     try! Data(pointer).write(to: imageEmbeddingsUrl)
     }
     */
    
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
    
    //MTLCaptureManager.shared().stopCapture()
    
    print(maskDecoderOutput.iou_predictions[0])
    print(maskDecoderOutput.iou_predictions[1])
    print(maskDecoderOutput.iou_predictions[2])
    print(maskDecoderOutput.iou_predictions[3])
  }
}

extension Array where Element == Point {
  func promptEncoderInput() -> PromptEncoderInput {
    let pointsShape = [1, count, 2]
    let labelsShape = [1, count]
    return PromptEncoderInput(
      points: MLShapedArray<Float>(
        scalars: self.map({
          [$0.x * (768.0 / 3024.0), $0.y * (1024.0 / 4032.0)]
        }).reduce(into: [],
                  {
                $0.append(contentsOf: $1)
                  }),
        shape: pointsShape
      ),
      labels: MLShapedArray<Float>(
        scalars: self.map({ Float($0.label) }),
        shape: labelsShape
      )
    )
  }
}
