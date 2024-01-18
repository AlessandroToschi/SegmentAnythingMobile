//
//  main.swift
//  SegmentAnythingMobileCLI
//
//  Created by Alessandro Toschi on 12/01/24.
//

import Foundation
import SAM
import Metal
import MetalKit

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let textureLoader = MTKTextureLoader(device: device)
let image = try! textureLoader.newTexture(
  URL: Bundle.main.url(
    forResource: "IMG_5102",
    withExtension: "jpg"
  )!,
  options: [
    .textureStorageMode: MTLStorageMode.shared.rawValue
  ]
)

/*
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = device
captureDescriptor.destination = .developerTools
try! MTLCaptureManager.shared().startCapture(with: captureDescriptor)

let commandBuffer = commandQueue.makeCommandBuffer()!

let preprocessing = Preprocessing(
  device: device,
  mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
  std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)
)
preprocessing.load()
let features = preprocessing.preprocess(
  image: image,
  commandBuffer: commandBuffer
)

MTLCaptureManager.shared().stopCapture()

Thread.sleep(forTimeInterval: 1.0)
*/

let startTime = DispatchTime.now()
let promptEncoder = PromptEncoder()
let promptEmbedding = promptEncoder.pointEmbeddings(
  points: [
    Point(x: 0.0, y: 0.0, label: 1),
    Point(x: 50.0, y: 50.0, label: 1),
    Point(x: 100.0, y: 100.0, label: 0)
  ],
  width: 1024,
  height: 1024
)
let endTime = DispatchTime.now()
let elapsed = (endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000
print("Elapsed \(elapsed) microseconds")
print(promptEmbedding.count)
