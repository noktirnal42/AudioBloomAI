import SwiftUI

@main
struct AudioBloomLiteApp: App {
    @StateObject private var audio = SimpleAudioCapture()
    @State private var isRunning = false

    var body: some Scene {
        WindowGroup {
            LiteContentView(audio: audio)
                .onAppear {
                    if !isRunning { try? audio.start(); isRunning = true }
                }
                .onDisappear { audio.stop(); isRunning = false }
        }
    }
}