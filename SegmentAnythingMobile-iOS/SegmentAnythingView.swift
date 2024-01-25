//
//  ContentView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 15/01/24.
//

import SwiftUI
import SAM
import PhotosUI

struct SegmentAnythingView: View {
  @State private var selectedPhotoItem: PhotosPickerItem? = nil
  @State private var selectedImageTexture: MTLTexture? = nil
  @State private var selectedImage: UIImage? = nil
  @State private var foregroundPoints: [Point] = []
  @State private var masks: [UIImage] = []
  @State private var maskMode: MaskProcessor.Mode = .addictive
  @State private var source: Int = 0
  
  let segmentAnything: SegmentAnything
  let maskProcessor: MaskProcessor
  let textureLoader: TextureLoader
  
  var maskModePicker: some View {
    Picker("Mask mode", selection: self.$maskMode) {
      Text("Add").tag(MaskProcessor.Mode.addictive)
      Text("Subtract").tag(MaskProcessor.Mode.subtractive)
    }
    .pickerStyle(.segmented)
  }
  
  var sourcePicker: some View {
    Picker("Source", selection: self.$source) {
      Text("Original").tag(0)
      if !self.masks.isEmpty {
        ForEach(1 ... self.masks.count, id: \.self) {
          Text("M\($0)").tag($0)
        }
      }
    }
    .pickerStyle(.segmented)
  }
  
  var resetButton: some View {
    Button("Reset") {
      self.foregroundPoints.removeAll()
    }
  }
  
  var selectPhotoLabel: some View {
    Text("Select a photo to start...")
  }
  
  var photoPicker: some View {
    PhotosPicker(
      selection: self.$selectedPhotoItem,
      matching: .any(of: [.images, .not(.livePhotos)])
    ) {
      Image(systemName: "camera")
    }
  }
    
  var body: some View {
    NavigationStack {
      Group {
        if let selectedImage {
          VStack {
            self.maskModePicker
            self.sourcePicker
            GeometryReader { geometry in
              ImageView(
                size: geometry.size,
                image: self.source == 0 ? selectedImage : self.masks[self.source - 1]
              ) { point in
                self.foregroundPoints.append(
                  Point(
                    x: Float(point.x),
                    y: Float(point.y),
                    label: 1
                  )
                )
              }
            }
            .clipped()
            self.resetButton
          }
          .padding()
        } else {
          self.selectPhotoLabel
        }
      }
      .navigationTitle("Segment Anything")
      .toolbar { self.photoPicker }
      .onChange(of: self.selectedPhotoItem) {
        _, _ in
        Task(priority: .high) {
          guard let selectedPhotoItem,
                let imageData = try? await selectedPhotoItem.loadTransferable(type: Data.self),
                let image = UIImage(data: imageData)
          else { return }
          
          self.foregroundPoints.removeAll()
          self.masks.removeAll()
          
          self.selectedImage = image
          self.selectedImageTexture = try! await self.textureLoader.loadTexture(uiImage: image)
          
          self.segmentAnything.preprocess(image: self.selectedImageTexture!)
        }
      }
      .onChange(of: self.foregroundPoints) {
        _, _ in
        guard !self.foregroundPoints.isEmpty
        else { return }
        Task(priority: .high) {
          self.masks = self.segmentAnything.predictMasks(
            points: self.foregroundPoints
          ).map { (mask, iou) in
            let maskTexture = self.maskProcessor.apply(
              input: self.selectedImageTexture!,
              mask: mask,
              mode: self.maskMode
            )
            return self.textureLoader.unloadTexture(texture: maskTexture)
          }
          print("ipppo")
        }
      }
    }
  }
}
