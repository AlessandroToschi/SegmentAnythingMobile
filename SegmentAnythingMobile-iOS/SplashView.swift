//
//  SplashView.swift
//  SegmentAnythingMobile-iOS
//
//  Created by Alessandro Toschi on 24/01/24.
//

import SwiftUI

struct SplashView: View {
  static private let title = "Segment Anything"
  
  let initializationAction: @Sendable () async -> ()
  
  @State private var titleOpacity: Double = 1.0
  
  var body: some View {
    VStack {
      Spacer()
      Text(SplashView.title)
        .font(.largeTitle)
        .opacity(self.titleOpacity)
        .transition(.opacity)
      Spacer()
      ProgressView().progressViewStyle(.circular).controlSize(.extraLarge)
    }.padding()

    .task(priority: .high, self.initializationAction)
    .task {
      try? await Task.sleep(nanoseconds: 500_000_000)
      withAnimation(Animation.easeOut(duration: 1.5).repeatForever(autoreverses: true)) {
        self.titleOpacity = 0.0
      }
    }
  }
}

#Preview {
  SplashView(
    initializationAction: {}
  )
}
