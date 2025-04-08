import SwiftUI
import AudioBloomCore
import CoreAudio
import Visualizer
import MLEngine

/// AudioBloom - Next generation audio visualizer for Apple Silicon
@main
struct AudioBloomApp: App {
    /// Application settings
    @StateObject private var settings = AudioBloomSettings()
    
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
                .environmentObject(settings)
                .environmentObject(audioEngine)
                .environmentObject(renderer)
                .environmentObject(mlProcessor)
                .onAppear {
                    // Initialize components on app launch
                    Task {
                        try? await initializeComponents()
                    }
                }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                // No new document functionality needed
            }
            
            // Custom menu commands
            CommandMenu("Visualization") {
                Menu("Theme") {
                    ForEach(AudioBloomCore.VisualTheme.allCases) { theme in
                        Button(theme.rawValue) {
                            settings.currentTheme = theme
                            updateVisualizationParameters()
                        }
                    }
                }
                
                Divider()
                
                Toggle("Enable Neural Engine", isOn: $settings.neuralEngineEnabled)
                    .onChange(of: settings.neuralEngineEnabled) { _ in
                        updateVisualizationParameters()
                    }
            }
        }
    }
    
    /// Initialize all application components
    private func initializeComponents() async throws {
        // Setup audio engine
        try await audioEngine.setupAudioSession()
        
        // Prepare renderer
        renderer.prepareRenderer()
        
        // Prepare ML model
        mlProcessor.prepareMLModel()
        
        // Set up initial visualization parameters
        updateVisualizationParameters()
        
        // Set up audio data connection to ML processor
        Task {
            while true {
                // We don't want to process every single audio frame, 
                // so we'll pause briefly between updates
                try? await Task.sleep(nanoseconds: 1_000_000_000 / UInt64(settings.frameRateTarget))
                
                // Only process if neural engine is enabled
                if settings.neuralEngineEnabled {
                    await mlProcessor.processAudioData(audioEngine.frequencyData)
                }
                
                // Update renderer with latest audio data
                renderer.update(
                    audioData: audioEngine.frequencyData,
                    levels: audioEngine.levels
                )
            }
        }
    }
    
    /// Updates visualization parameters based on current settings
    private func updateVisualizationParameters() {
        if let parameterReceiver = renderer as? VisualizationParameterReceiver {
            parameterReceiver.updateParameters(settings.visualizationParameters())
        }
    }
}

