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
  @State private var isLoading = false
  
  let segmentAnything: SegmentAnything
  let maskProcessor: MaskProcessor
  let textureLoader: TextureLoader
  
  var maskModePicker: some View {
    Picker("Mask mode", selection: self.$maskMode) {
      Text("Add").tag(MaskProcessor.Mode.addictive)
      Text("Subtract").tag(MaskProcessor.Mode.subtractive)
    }
    .pickerStyle(.segmented)
    .disabled(self.masks.isEmpty)
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
    .disabled(self.masks.isEmpty)
  }
  
  var resetButton: some View {
    Button("Reset", action: self.reset)
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
  
  var draggablePin: some View {
    HStack {
      Image(systemName: "pin")
        .resizable()
        .scaledToFit()
        .frame(width: 32.0, height: 32.0)
        .draggable("pin", preview: { EmptyView() })
        .padding(.trailing, 20)
      Text("Drop the pin into the image point you want to segment.")
      Spacer()
    }
  }
  
  var body: some View {
    NavigationStack {
      Group {
        VStack {
          if let selectedImage {
            self.maskModePicker
            self.sourcePicker
            self.draggablePin
            GeometryReader { geometry in
              ImageView(
                size: geometry.size,
                image: self.source == 0 ? selectedImage : self.masks[self.source - 1],
                foregroundPoints: self.$foregroundPoints
              )
            }
            .clipped()
            self.resetButton
          }
          else {
            Spacer()
            self.selectPhotoLabel
            Spacer()
          }
        }
      }
      .padding([.leading, .trailing])
      .overlay() {
        if self.isLoading {
          LoadingView()
        }
      }
      .navigationTitle("Segment Anything")
      .toolbar { self.photoPicker }
      .onChange(of: self.selectedPhotoItem) {
        _, _ in
        Task(priority: .high) {
          self.isLoading = true
          guard let selectedPhotoItem,
                let imageData = try? await selectedPhotoItem.loadTransferable(type: Data.self),
                let image = UIImage(data: imageData)
          else { return }
          
          self.reset()
          
          self.selectedImage = image
          self.selectedImageTexture = try! await self.textureLoader.loadTexture(uiImage: image)
          
          self.segmentAnything.preprocess(image: self.selectedImageTexture!)
          self.isLoading = false
        }
      }
      .onChange(of: self.foregroundPoints) {
        _, _ in
        guard !self.foregroundPoints.isEmpty
        else { return }
#if targetEnvironment(simulator)
        return
#endif
        Task(priority: .high, operation: self.predictMasks)
      }
      .onChange(of: self.maskMode) {
        _, _ in
        guard !self.foregroundPoints.isEmpty,
              !self.masks.isEmpty
        else { return }
        Task(priority: .high, operation: self.predictMasks)
      }
    }
  }
  
  func reset() {
    self.foregroundPoints.removeAll()
    self.masks.removeAll()
  }
  
  @Sendable
  func predictMasks() async {
    self.isLoading = true
    self.masks = self.segmentAnything.predictMasks(
      points: self.foregroundPoints
    ).map {
      (mask, iou) in
      let maskTexture = self.maskProcessor.apply(
        input: self.selectedImageTexture!,
        mask: mask,
        mode: self.maskMode
      )
      return self.textureLoader.unloadTexture(texture: maskTexture)
    }
    self.isLoading = false
  }
}
