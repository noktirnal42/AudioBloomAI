import SwiftUI
import AudioBloomCore
import Visualizer
import MLEngine
import AudioProcessor
@main
public struct AudioBloomApp: App {
    /// Application settings
    @StateObject private var settings = AudioBloomSettings()
    
    /// The state object that manages our audio engine
    @StateObject private var audioEngine = AudioEngine()
    
    /// The state object that manages our Metal renderer
    @StateObject private var renderer = MetalRenderer()
    
    /// The state object that manages our ML processor
    @StateObject private var mlProcessor = MLProcessor()
    
    public init() {
        // Prepare core components
        renderer.prepareRenderer()
        mlProcessor.prepareMLModel()
        
        // Initialize audio engine and set up the audio session
        Task {
            try? await audioEngine.setupAudioSession()
        }
    }
    
    public var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(audioEngine)
                .environmentObject(renderer)
                .environmentObject(mlProcessor)
        }
        .windowStyle(.hiddenTitleBar) // Hides the title bar on macOS
    }
}
