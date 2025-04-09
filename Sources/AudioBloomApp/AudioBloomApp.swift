import SwiftUI
import AudioProcessor
import MLEngine
import Visualizer
import AudioBloomCore
import Combine

@main
struct AudioBloomApp: App {
    // MARK: - State Objects
    
    /// Audio engine for capturing and processing audio
    @StateObject private var audioEngine = AudioEngine()
    
    /// ML processor for analyzing audio with Neural Engine
    @StateObject private var mlProcessor = MLProcessor(optimizationLevel: .balanced)
    
    /// Audio bridge connecting the audio engine to the ML processor
    @StateObject private var audioBridge: AudioBridge
    
    /// Metal renderer for visualization
    @StateObject private var renderer = MetalRenderer()
    
    /// Application settings
    @StateObject private var settings = AudioBloomSettings()
    
    /// Cancellables for managing subscriptions
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    init() {
        // Initialize the audio bridge with our ML processor
        let bridge = AudioBridge(mlProcessor: mlProcessor)
        _audioBridge = StateObject(wrappedValue: bridge)
        
        // Apply initial settings
        applyInitialSettings()
    }
    
    // MARK: - App Scene
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                // Inject all environment objects
                .environmentObject(audioEngine)
                .environmentObject(mlProcessor)
                .environmentObject(audioBridge)
                .environmentObject(renderer)
                .environmentObject(settings)
                .onAppear {
                    setupAudioSystem()
                }
                .onDisappear {
                    teardownAudioSystem()
                }
        }
        .windowStyle(.hiddenTitleBar) // Hides the title bar on macOS
    }
    
    // MARK: - Setup and Teardown
    
    /// Sets up the audio processing system and visualization pipeline
    private func setupAudioSystem() {
        // Start by preparing the renderer
        Task { @MainActor in
            renderer.prepareRenderer()
        }
        
        // Setup neural engine audio pipeline
        Task {
            do {
                // Setup audio session
                try await audioEngine.setupAudioSession()
                
                // Connect the visualization data from the bridge to the renderer
                audioBridge.visualizationPublisher
                    .receive(on: RunLoop.main)
                    .sink { [weak renderer] data in
                        renderer?.updateVisualization(with: data)
                    }
                    .store(in: &cancellables)
                
                // Connect audio bridge to audio engine
                audioBridge.connect(to: audioEngine)
                
                // Activate the bridge to start ML processing
                audioBridge.activate()
                
                // Start audio capture
                try audioEngine.startCapture()
                
                // Setup performance monitoring
                setupPerformanceMonitoring()
            } catch {
                print("Failed to setup audio system: \(error)")
            }
        }
        
        // Subscribe to settings changes
        settings.$neuralEngineEnabled
            .sink { [weak mlProcessor] enabled in
                mlProcessor?.useNeuralEngine = enabled
            }
            .store(in: &cancellables)
        
        settings.$currentTheme
            .sink { [weak renderer] theme in
                renderer?.updateTheme(theme)
            }
            .store(in: &cancellables)
    }
    
    /// Tears down the audio processing system
    private func teardownAudioSystem() {
        // Stop audio capture
        audioEngine.stopCapture()
        
        // Deactivate bridge
        audioBridge.deactivate()
        
        // Disconnect bridge
        audioBridge.disconnect()
        
        // Clean up subscriptions
        cancellables.removeAll()
        
        // Clean up ML processor resources
        mlProcessor.cleanup()
        
        // Clean up renderer resources
        renderer.cleanup()
    }
    
    /// Setup performance monitoring
    private func setupPerformanceMonitoring() {
        // Monitor ML processor performance
        mlProcessor.delegate = PerformanceMonitor.shared
    }
    
    /// Apply initial settings
    private func applyInitialSettings() {
        // Set Neural Engine usage based on settings
        mlProcessor.useNeuralEngine = settings.neuralEngineEnabled
        
        // Apply optimization level based on device capabilities
        #if os(macOS)
        mlProcessor.optimizationLevel = .balanced
        #else
        // Check for high-end devices vs. battery-sensitive devices
        if ProcessInfo.processInfo.thermalState == .nominal {
            mlProcessor.optimizationLevel = .performance
        } else {
            mlProcessor.optimizationLevel = .efficiency
        }
        #endif
    }
}

/// Shared performance monitor
class PerformanceMonitor: MLProcessingDelegate {
    /// Shared instance
    static let shared = PerformanceMonitor()
    
    /// Latest performance metrics
    @Published var latestMetrics: MLProcessor.PerformanceMetrics?
    
    // MARK: - MLProcessingDelegate
    
    func mlProcessorReadyStateChanged(isReady: Bool) {
        print("ML Processor ready state changed: \(isReady)")
    }
    
    func mlProcessorDidEncounterError(_ error: Error) {
        print("ML Processor error: \(error.localizedDescription)")
    }
    
    func mlProcessorDidUpdateMetrics(_ metrics: MLProcessor.PerformanceMetrics) {
        self.latestMetrics = metrics
        
        // Log occasional performance data
        if arc4random_uniform(100) < 5 { // ~5% of updates
            print("Neural Engine: \(Int(metrics.neuralEngineUtilization * 100))% | FPS: \(Int(metrics.framesPerSecond))")
        }
    }
}
