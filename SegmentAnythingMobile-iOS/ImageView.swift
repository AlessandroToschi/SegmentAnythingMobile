//
//  ImageView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 24/01/24.
//

import Foundation
import SwiftUI

struct ImageView: View {
  let size: CGSize
  let image: UIImage
  let onLongPressAction: (CGPoint) -> ()
  
  @State private var initialScale: CGPoint = CGPoint.zero
  @State private var scale: CGPoint = CGPoint.zero
  @State private var traslation: CGPoint = CGPoint.zero
  
  @State private var lastTouch: CGPoint?
  @State private var lastScale: CGFloat?
  @State private var longPressTouch: CGPoint = .zero
  
  init(
    size: CGSize,
    image: UIImage,
    onLongPressAction: @escaping (CGPoint) -> Void
  ) {
    self.size = size
    self.image = image
    self.onLongPressAction = onLongPressAction

    let scale = min(
      self.size.width / self.image.size.width,
      self.size.height / self.image.size.height
    )
    self._scale = State(
      initialValue: CGPoint(
        x: scale,
        y: scale
      )
    )
    self._initialScale = State(initialValue: self.scale)
    self._traslation = State(
      initialValue: CGPoint(
        x: (self.size.width - (self.image.size.width * scale)) / 2.0,
        y: (self.size.height - (self.image.size.height * scale)) / 2.0
      )
    )
  }
  
  
  private var currentTransform: CGAffineTransform {
    CGAffineTransform(
      self.scale.x,
      0.0,
      0.0,
      self.scale.y,
      self.traslation.x,
      self.traslation.y
    )
  }
  
  private var dragGesture: some Gesture {
    DragGesture(minimumDistance: 0.0)
      .onChanged({
        value in
        
        if self.lastTouch == nil {
          self.lastTouch = value.translation.asPoint()
        }
        
        self.traslation.x += value.translation.width - self.lastTouch!.x
        self.traslation.y += value.translation.height - self.lastTouch!.y
        
        self.lastTouch = value.translation.asPoint()
        
        self.longPressTouch = value.location
      })
      .onEnded({ _ in self.lastTouch = nil })
  }
  
  private var longPressGesture: some Gesture {
    LongPressGesture(minimumDuration: 1.0)
      .onEnded { _ in
        let currentTransform = self.currentTransform
        let imageRect = CGRect(origin: .zero, size: self.image.size)
        let transformedImageRect = imageRect.applying(currentTransform)
        
        guard transformedImageRect.contains(self.longPressTouch)
        else { return }
        
        let imageLongPressPoint = self.longPressTouch.applying(currentTransform.inverted())
        
        print(imageLongPressPoint)
        
        self.onLongPressAction(imageLongPressPoint)
      }
  }
  
  private var scaleGesture: some Gesture {
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
        
        var scaleTransform = CGAffineTransformIdentity
        scaleTransform = scaleTransform.translatedBy(x: anchorPoint.x, y: anchorPoint.y)
        scaleTransform = scaleTransform.scaledBy(x: 1.0 + delta, y: 1.0 + delta)
        scaleTransform = scaleTransform.translatedBy(x: -anchorPoint.x, y: -anchorPoint.y)
        
        let newTransform = self.currentTransform.concatenating(scaleTransform)
        
        self.scale.x = newTransform.a
        self.scale.y = newTransform.d
        self.traslation.x = newTransform.tx
        self.traslation.y = newTransform.ty
        
        self.lastScale = value.magnification
      })
      .onEnded({ _ in self.lastScale = nil })
  }
  
  var body: some View {
    Canvas {
      graphicsContext, size in
      
      //graphicsContext.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.gray))
      
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
    .gesture(self.longPressGesture.simultaneously(with: self.dragGesture))
    .gesture(self.scaleGesture)
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
