import SwiftUI
import AudioBloomCore
import CoreAudio
import Visualizer
import MLEngine

/// AudioBloom - Next generation audio visualizer for Apple Silicon
/// Main application entry point
@main
struct AudioBloomApp: App {
    /// The state object that manages our audio engine
    @StateObject private var audioEngine = AudioEngine()
    
    /// The state object that manages our Metal renderer
    @StateObject private var renderer = MetalRenderer()
    
    /// The state object that manages ML processing
    @StateObject private var mlProcessor = MLProcessor()
    
    /// Application body - defines the main window and views
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(audioEngine)
                .environmentObject(renderer)
                .environmentObject(mlProcessor)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            // Custom menu commands will go here
            CommandGroup(replacing: .newItem) {
                // No new document functionality needed
            }
        }
    }
}

/// Main content view for the application
struct ContentView: View {
    /// Reference to the audio engine
    @EnvironmentObject var audioEngine: AudioEngine
    
    /// Reference to the Metal renderer
    @EnvironmentObject var renderer: MetalRenderer
    
    /// Reference to the ML processor
    @EnvironmentObject var mlProcessor: MLProcessor
    
    /// Visualization state
    @State private var isPlaying = false
    
    var body: some View {
        VStack {
            VisualizerView()
                .environmentObject(audioEngine)
                .environmentObject(renderer)
                .environmentObject(mlProcessor)
            
            ControlPanel(isPlaying: $isPlaying)
        }
        .onAppear {
            // Initialize components
            Task {
                await audioEngine.setupAudioSession()
            }
            renderer.prepareRenderer()
            mlProcessor.prepareMLModel()
        }
    }
}

/// Placeholder visualizer view that will host our Metal content
struct VisualizerView: View {
    @EnvironmentObject var renderer: MetalRenderer
    
    var body: some View {
        ZStack {
            // This will be replaced with a MetalView
            Color.black
                .edgesIgnoringSafeArea(.all)
            Text("AudioBloom Visualizer")
                .foregroundColor(.white)
        }
    }
}

/// Control panel for audio visualization
struct ControlPanel: View {
    @Binding var isPlaying: Bool
    
    var body: some View {
        HStack {
            Button(action: {
                isPlaying.toggle()
            }) {
                Image(systemName: isPlaying ? "pause.circle" : "play.circle")
                    .font(.largeTitle)
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            // Additional controls will go here
        }
        .padding()
    }
}

