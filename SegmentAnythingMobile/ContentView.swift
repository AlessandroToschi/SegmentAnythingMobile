//
//  ContentView.swift
//  SegmentAnythingMobile
//
//  Created by Alessandro Toschi on 10/01/24.
//

import SwiftUI
import SAM
import Metal
import MetalKit

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .onAppear() {
          let device = MTLCreateSystemDefaultDevice()!
          let textureLoader = MTKTextureLoader(device: device)
          let texture = try! textureLoader.newTexture(
            URL: Bundle.main.url(
              forResource: "IMG_5102",
              withExtension: "jpg"
            )!
          )

          let sam = SegmentAnythingMobile(device: device)
          sam.predict(image: texture)
        }
        .padding()
    }
}
