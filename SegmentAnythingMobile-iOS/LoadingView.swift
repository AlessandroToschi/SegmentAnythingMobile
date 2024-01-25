//
//  LoadingView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 25/01/24.
//

import SwiftUI

struct LoadingView: View {
  @State private var opacity = 0.0
  
  var body: some View {
    Color.clear
      .background(.ultraThinMaterial)
      .overlay() {
        ProgressView()
          .controlSize(.extraLarge)
      }
      .opacity(self.opacity)
      .transition(.opacity)
      .ignoresSafeArea()
      .onAppear() {
        withAnimation(Animation.easeIn(duration: 0.25)) {
          self.opacity = 1.0
        }
      }
  }
}

#Preview {
  LoadingView()
}

