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

struct ContentView: View {
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
              )
              
            }.coordinateSpace(name: "canvas")
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

struct ImageView: View {
  let size: CGSize
  let image: UIImage
  
  @State var initialScale: CGSize = CGSize(width: 1.0, height: 1.0)
  @State var scale: CGSize = CGSize(width: 1.0, height: 1.0)
  @State var offset: CGSize = CGSize.zero
  
  @State var lastTouch: CGSize?
  @State var lastScale: CGFloat?
  @State var anchorPoint: UnitPoint?
  
  var body: some View {
    Image(uiImage: image)
      .frame(width: self.image.size.width, height: self.image.size.height)
      .scaleEffect(
        self.scale,
        anchor: .topLeading
      )
      .offset(self.offset)
      .gesture(
        DragGesture()
          .onChanged({
            value in
            if self.lastTouch == nil {
              self.lastTouch = value.translation
            }
            self.offset.width += value.translation.width - self.lastTouch!.width
            self.offset.height += value.translation.height - self.lastTouch!.height
            self.lastTouch = value.translation
          })
          .onEnded({ _ in
            self.lastTouch = nil
          })
      )
      .gesture(
        MagnifyGesture()
          .onChanged({
            value in
            if self.lastScale == nil {
              self.lastScale = value.magnification
            }
            if self.anchorPoint == nil {
              self.anchorPoint = value.startAnchor
            }
            let delta = value.magnification - self.lastScale!
            self.scale.width += delta
            self.scale.height += delta
                        
            self.scale.width = min(
              5.0 * self.initialScale.width,
              max(self.initialScale.width, self.scale.width)
            )
            self.scale.height = min(
              5.0 * self.initialScale.height,
              max(self.initialScale.height, self.scale.height)
            )

            self.lastScale = value.magnification
          })
          .onEnded({_ in
            self.lastScale = nil
            self.anchorPoint = nil
          })
      )
      .onTapGesture(count: 2) {
        self.reset()
      }
      .onChange(of: self.size) {
        _, _ in
        self.reset()
      }
      .onChange(of: self.image) {
        _, _ in
        self.reset()
      }
      .onAppear() {
        self.reset()
      }
  }
  
  func reset() {
    let scale = min(
      self.size.width / self.image.size.width,
      self.size.height / self.image.size.height
    )
    self.scale = CGSize(
      width: scale, //self.size.width / self.image.size.width,
      height: scale //self.size.height / self.image.size.height
    )
    self.initialScale = self.scale
    self.offset = CGSize(
      width: (self.size.width - (self.image.size.width * scale)) / 2.0,
      height: (self.size.height - (self.image.size.height * scale)) / 2.0
    )
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
