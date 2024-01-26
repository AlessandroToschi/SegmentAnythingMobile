//
//  ImageView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 24/01/24.
//

import Foundation
import SwiftUI
import SAM

struct ImageView: View {
  let size: CGSize
  let image: UIImage
  @Binding var foregroundPoints: [Point]
  
  @State private var initialScale: CGPoint = CGPoint.zero
  @State private var scale: CGPoint = CGPoint.zero
  @State private var traslation: CGPoint = CGPoint.zero
  
  @State private var lastTouch: CGPoint?
  @State private var lastScale: CGFloat?
  
  
  init(
    size: CGSize,
    image: UIImage,
    foregroundPoints: Binding<[Point]>
  ) {
    self.size = size
    self.image = image
    self._foregroundPoints = foregroundPoints

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
      })
      .onEnded({ _ in self.lastTouch = nil })
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
            
      graphicsContext.translateBy(x: self.traslation.x, y: self.traslation.y)
      graphicsContext.scaleBy(x: self.scale.x, y: self.scale.y)
      
      graphicsContext.draw(Image(uiImage: self.image), at: .zero, anchor: .topLeading)
      
      let circleRadius = 8.0 / self.scale.x
      
      for foregroundPoint in self.foregroundPoints {
        let circleOrigin = CGPoint(
          x: Double(foregroundPoint.x) - circleRadius,
          y: Double(foregroundPoint.y) - circleRadius
        )
        graphicsContext.fill(
          Path(
            ellipseIn: CGRect(
              origin: circleOrigin,
              size: CGSize(
                width: circleRadius * 2.0,
                height: circleRadius * 2.0
              )
            )
          ),
          with: .color(.blue)
        )
      }
    }
    .id(self.image)
    .id(self.size)
    .onChange(of: self.image) { _, _ in self.reset() }
    .onChange(of: self.size) { _, _ in self.reset() }
    .onAppear(perform: self.reset)
    .onTapGesture(count: 2, perform: self.reset)
    .gesture(self.dragGesture)
    .gesture(self.scaleGesture)
    .dropDestination(for: String.self) {
      _, location in
      
      let currentTransform = self.currentTransform
      let imageRect = CGRect(origin: .zero, size: self.image.size)
      let transformedImageRect = imageRect.applying(currentTransform)
      
      guard transformedImageRect.contains(location)
      else { return true }
      
      let imagePinPoint = location.applying(currentTransform.inverted())
      
      self.foregroundPoints.append(
        Point(
          x: Float(imagePinPoint.x),
          y: Float(imagePinPoint.y),
          label: 1
        )
      )
      
      return true
    }
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
