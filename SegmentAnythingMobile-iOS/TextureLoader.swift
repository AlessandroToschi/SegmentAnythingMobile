//
//  TextureLoader.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 25/01/24.
//

import Foundation
import Metal
import MetalKit

class TextureLoader {
  private let ciContext: CIContext
  private let textureLoader: MTKTextureLoader
  
  init(device: MTLDevice) {
    self.ciContext = CIContext(mtlDevice: device)
    self.textureLoader = MTKTextureLoader(device: device)
  }
  
  func loadTexture(uiImage: UIImage) async throws -> MTLTexture {
    guard let ciImage = CIImage(
      image: uiImage,
      options: [
        .applyOrientationProperty: true,
        .properties: [kCGImagePropertyOrientation: CGImagePropertyOrientation(uiImage.imageOrientation).rawValue]
      ]
    ), let cgImage = self.ciContext.createCGImage(ciImage, from: ciImage.extent)
    else { throw NSError() }
    
    return try await self.textureLoader.newTexture(
      cgImage: cgImage,
      options: [
        .textureStorageMode: MTLStorageMode.shared.rawValue
      ]
    )
  }
  
  func unloadTexture(texture: MTLTexture) -> UIImage {
    let ciImage = CIImage(mtlTexture: texture)!
    let transform = CGAffineTransform.identity.scaledBy(x: 1, y: -1).translatedBy(x: 0, y: ciImage.extent.height)
    let transformed = ciImage.transformed(by: transform)
    
    let cgImage = self.ciContext.createCGImage(transformed, from: transformed.extent)!
    return UIImage(cgImage: cgImage)
  }
}

fileprivate extension CGImagePropertyOrientation {
  init(_ uiOrientation: UIImage.Orientation) {
    switch uiOrientation {
      case .up: self = .up
      case .upMirrored: self = .upMirrored
      case .down: self = .down
      case .downMirrored: self = .downMirrored
      case .left: self = .left
      case .leftMirrored: self = .leftMirrored
      case .right: self = .right
      case .rightMirrored: self = .rightMirrored
      @unknown default: self = .up
    }
  }
}
