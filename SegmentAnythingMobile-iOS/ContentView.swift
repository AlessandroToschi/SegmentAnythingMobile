//
//  ContentView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 15/01/24.
//

import SwiftUI
import SAM
import MetalKit

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .padding()
        .onAppear() {
          let device = MTLCreateSystemDefaultDevice()!
          let commandQueue = device.makeCommandQueue()!

          let textureLoader = MTKTextureLoader(device: device)
          let image = try! textureLoader.newTexture(
            name: "IMG_5102",
            scaleFactor: 1.0,
            bundle: Bundle.main
          )
          
          let commandBuffer = commandQueue.makeCommandBuffer()!

          let preprocessing = Preprocessing(
            device: device,
            mean: SIMD3<Float>(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
            std: SIMD3<Float>(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)
          )
          preprocessing.preprocess(
            image: image,
            commandBuffer: commandBuffer
          )
        }
    }
}
