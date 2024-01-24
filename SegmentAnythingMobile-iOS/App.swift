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
  
  var body: some Scene {
    WindowGroup {
      if !self.initializationCompleted {
        SplashView(initializationAction: self.initialize)
      } else {
        SegmentAnythingView()
      }
    }
  }
  
  @Sendable
  func initialize() {
    guard let device = MTLCreateSystemDefaultDevice()
    else { return }
    
    self.segmentAnything = SegmentAnything(device: device)
    self.segmentAnything.load()
    
    self.initializationCompleted = true
  }
}
