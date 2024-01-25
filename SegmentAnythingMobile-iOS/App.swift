//
//  SegmentAnythingMobile_iOSApp.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 15/01/24.
//

import SwiftUI
import SAM
import Metal

@main
struct SAMApp: App {
  @State private var initializationCompleted: Bool = false
  @State private var segmentAnything: SegmentAnything!
  @State private var maskProcessor: MaskProcessor!
  @State private var textureLoader: TextureLoader!
  
  var body: some Scene {
    WindowGroup {
      if !self.initializationCompleted {
        SplashView(initializationAction: self.initialize)
      } else {
        SegmentAnythingView(
          segmentAnything: self.segmentAnything,
          maskProcessor: self.maskProcessor,
          textureLoader: self.textureLoader
        )
      }
    }
  }
  
  @Sendable
  func initialize() {
    guard let device = MTLCreateSystemDefaultDevice()
    else { return }
    
    self.segmentAnything = SegmentAnything(device: device)
    self.segmentAnything.load()
    
    self.maskProcessor = MaskProcessor(device: device)
    self.maskProcessor.load()
    
    self.textureLoader = TextureLoader(device: device)
    
    self.initializationCompleted = true
  }
}
