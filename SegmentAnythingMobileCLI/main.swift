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

let segmentAnything = SegmentAnything(device: device)
segmentAnything.load()
segmentAnything.preprocess(image: image)
let startTime = DispatchTime.now()


segmentAnything.predictMasks(
  points: [
    Point(
      x: 1526,
      y: 2456,
      label: 1
    ),
    Point(
      x: 1833,
      y: 2397,
      label: 1
    )
  ]
)

let endTime = DispatchTime.now()
let elapsed = (endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000
print("Elapsed \(elapsed) ms")
