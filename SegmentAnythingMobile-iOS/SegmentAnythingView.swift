//
//  ContentView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 15/01/24.
//

import SwiftUI
import SAM
import MetalKit
import PhotosUI

struct SegmentAnythingView: View {
  @State var selectedPhotoItem: PhotosPickerItem? = nil
  @State var selectedImage: UIImage? = nil
  
  var body: some View {
    NavigationStack {
      VStack {
        Group {
          if let selectedImage {
            GeometryReader { geometry in
              ImageView(
                size: geometry.size,
                image: selectedImage
              ) { _ in
                
              }
            }
          } else {
            Text("Select a photo to start...")
          }
        }
        .padding()
        .clipped()
      }
      .navigationTitle("Segment Anything")
      .toolbar {
        PhotosPicker(
          selection: self.$selectedPhotoItem,
          matching: .any(of: [.images, .not(.livePhotos)])
        ) {
          Image(systemName: "camera")
        }
      }
      .onChange(of: self.selectedPhotoItem) {
        _, _ in
        Task(priority: .background) {
          guard let selectedPhotoItem,
                let imageData = try? await selectedPhotoItem.loadTransferable(type: Data.self),
                let image = UIImage(data: imageData)
          else { return }
          self.selectedImage = image
        }
      }
    }
  }
}

/*
 .onAppear() {
 let device = MTLCreateSystemDefaultDevice()!
 let commandQueue = device.makeCommandQueue()!
 
 let textureLoader = MTKTextureLoader(device: device)
 let image = try! textureLoader.newTexture(
 name: "IMG_5102",
 scaleFactor: 1.0,
 bundle: Bundle.main
 )
 }
 */

struct ContentView_Preview: PreviewProvider {
  static var previews: some View {
    SegmentAnythingView()
  }
}


