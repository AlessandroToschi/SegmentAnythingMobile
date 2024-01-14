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
let textureLoader = MTKTextureLoader(device: device)
let texture = try! textureLoader.newTexture(
  URL: Bundle.main.url(
    forResource: "IMG_5102",
    withExtension: "jpg"
  )!,
  options: [
    .textureStorageMode: MTLStorageMode.shared.rawValue
  ]
)

let sam = SegmentAnythingMobile(device: device)
sam.predict(image: texture)
