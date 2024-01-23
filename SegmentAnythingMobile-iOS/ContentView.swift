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
              CanvasView(
                size: geometry.size,
                image: selectedImage
              )
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

struct CanvasView: View {
  let size: CGSize
  let image: UIImage
  
  @State private var initialScale: CGPoint = CGPoint.zero
  @State private var scale: CGPoint = CGPoint.zero
  @State private var traslation: CGPoint = CGPoint.zero
  
  @State private var lastTouch: CGPoint?
  @State private var lastScale: CGFloat?
  
  var body: some View {
    Canvas {
      graphicsContext, size in
      
      graphicsContext.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.gray))
            
      graphicsContext.translateBy(x: self.traslation.x, y: self.traslation.y)
      graphicsContext.scaleBy(x: self.scale.x, y: self.scale.y)
      
      graphicsContext.draw(Image(uiImage: self.image), at: .zero, anchor: .topLeading)
    }
    .id(self.image)
    .id(self.size)
    .onChange(of: self.image) { _, _ in self.reset() }
    .onChange(of: self.size) { _, _ in self.reset() }
    .onAppear(perform: self.reset)
    .onTapGesture(count: 2, perform: self.reset)
    .gesture(
      DragGesture()
        .onChanged({
          value in
          if self.lastTouch == nil {
            self.lastTouch = value.translation.asPoint()
          }
          self.traslation.x += value.translation.width - self.lastTouch!.x
          self.traslation.y += value.translation.height - self.lastTouch!.y
          self.lastTouch = value.translation.asPoint()
        })
        .onEnded({ _ in self.lastTouch = nil })
    )
    .gesture(
      MagnifyGesture()
        .onChanged({
          value in
          if self.lastScale == nil {
            self.lastScale = value.magnification
          }
        
          let anchorPoint = CGPoint(
            x: self.size.width * value.startAnchor.x,
            y: self.size.height * value.startAnchor.y
          )
          
          let delta = value.magnification - self.lastScale!
          
          let currentTransform = CGAffineTransform(
            a: self.scale.x,
            b: 0.0,
            c: 0.0,
            d: self.scale.y,
            tx: self.traslation.x,
            ty: self.traslation.y
          )
          
          var scaleTransform = CGAffineTransformIdentity
          scaleTransform = scaleTransform.translatedBy(x: anchorPoint.x, y: anchorPoint.y)
          scaleTransform = scaleTransform.scaledBy(x: 1.0 + delta, y: 1.0 + delta)
          scaleTransform = scaleTransform.translatedBy(x: -anchorPoint.x, y: -anchorPoint.y)
          
          let newTransform = currentTransform.concatenating(scaleTransform)
          
          self.scale.x = newTransform.a
          self.scale.y = newTransform.d
          self.traslation.x = newTransform.tx
          self.traslation.y = newTransform.ty
          
          self.lastScale = value.magnification
        })
        .onEnded({ _ in self.lastScale = nil })
    )
  }
  
  func reset() {
    let scale = min(
      self.size.width / self.image.size.width,
      self.size.height / self.image.size.height
    )
    self.scale = CGPoint(
      x: scale,
      y: scale
    )
    self.initialScale = self.scale
    self.traslation = CGPoint(
      x: (self.size.width - (self.image.size.width * scale)) / 2.0,
      y: (self.size.height - (self.image.size.height * scale)) / 2.0
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

extension CGSize: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(self.width)
    hasher.combine(self.height)
  }
  
  public func asPoint() -> CGPoint {
    return CGPoint(
      x: self.width,
      y: self.height
    )
  }
}
