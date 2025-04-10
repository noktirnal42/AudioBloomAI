// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
import SwiftUI
import Combine
import AudioBloomCore
import AudioProcessor
import Visualizer
import MLEngine
import AudioBloomUI

/// Main application for AudioBloomAI
@main
@available(macOS 15.0, *)
struct AudioBloomApp: Sendable: App {
    /// App delegate for handling app lifecycle events
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    /// Environment object for app-wide state
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .onAppear {
                    Task {
                        await appState.initialize()
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About AudioBloomAI") {
                    appState.showAbout = true
                }
            }
            
            CommandGroup(after: .appVisibility) {
                Button("Audio Settings") {
                    appState.showAudioSettings = true
                }
                .keyboardShortcut("a", modifiers: [.command, .option])
                
                Button("Visualization Settings") {
                    appState.showVisualizerSettings = true
                }
                .keyboardShortcut("v", modifiers: [.command, .option])
            }
            
            CommandGroup(after: .windowSize) {
                Button("Enter Full Screen") {
                    if let window = NSApp.windows.first {
                        window.toggleFullScreen(nil)
                    }
                }
                .keyboardShortcut("f", modifiers: [.command])
            }
        }
    }
}

/// App Delegate for handling app lifecycle
@available(macOS 15.0, *)
@MainActor
class AppDelegate: NSObject, NSApplicationDelegate  {
    // Added @MainActor for UI thread safety
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set up any required application-level configurations
        setupAppAppearance()
    }
    
    func applicationWillTerminate(_ notification: Notification) {
        // Clean up resources when the app terminates
        AudioBloomCore.AppSettings.shared.save()
    }
    
    private func setupAppAppearance() {
        // Configure app appearance
        let appearance = NSAppearance(named: .darkAqua)
        NSApp.appearance = appearance
    }
}

/// Application-wide state management
@available(macOS 15.0, *)
@MainActor
class AppState: ObservableObject  {
    // Added @MainActor for UI thread safety
    /// Audio engine for processing audio
    @Published private(set) var audioEngine: AudioEngine?
    
    /// Audio visualizer for rendering
    @Published private(set) var visualizer: AudioVisualizer?
    
    /// Neural engine for ML-based audio analysis
    @Published private(set) var neuralEngine: NeuralEngine?
    
    /// Application settings
    private let settings = AudioBloomCore.AppSettings.shared
    
    /// UI state management
    @Published var showAbout = false
    @Published var showAudioSettings = false
    @Published var showVisualizerSettings = false
    
    /// Error handling
    @Published var errorMessage: String?
    @Published var showError = false
    
    /// State management
    @Published var isInitialized = false
    @Published var isProcessingAudio = false
    
    /// Selected visualization mode
    @Published var selectedVisualizationMode: VisualizationMode = .spectrum {
        didSet {
            visualizer?.setVisualizationMode(selectedVisualizationMode)
        }
    }
    
    /// Selected theme
    @Published var selectedTheme: Theme = .classic {
        didSet {
            visualizer?.setTheme(selectedTheme)
        }
    }
    
    /// Audio sensitivity
    @Published var sensitivity: Float = 0.8 {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Motion intensity
    @Published var motionIntensity: Float = 0.7 {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Color intensity
    @Published var colorIntensity: Float = 0.8 {
        didSet {
            updateVisualizationParameters()
        }
    }
    
    /// Neural processing enabled flag
    @Published var neuralProcessingEnabled = false {
        didSet {
            if neuralProcessingEnabled {
                initializeNeuralEngine()
            } else {
                neuralEngine = nil
            }
        }
    }
    
    /// Subscription store
    private var cancellables = Set<AnyCancellable>()
    
    /// Initialize the app components
    func initialize() async {
        // Create and configure components
        do {
            // Initialize audio engine
            let engine = AudioEngine()
            audioEngine = engine
            
            // Initialize visualizer
            visualizer = AudioVisualizer()
            
            // Load settings
            loadSettings()
            
            // Connect components
            visualizer?.connectToAudioProcessor(engine)
            
            // Set up audio session
            try await engine.setupAudioSession()
            
            // Request permission to access microphone
            if await requestMicrophoneAccess() {
                // Start audio processing
                try await startAudioProcessing()
            } else {
                handleError("Microphone access denied. Please enable in System Settings.")
            }
            
            Task { @MainActor in
                self.isInitialized = true
            }
            
        } catch {
            handleError("Failed to initialize: \(error.localizedDescription)")
        }
    }
    
    /// Start audio processing
    func startAudioProcessing() async throws {
        do {
            try audioEngine?.startCapture()
            Task { @MainActor in
                self.isProcessingAudio = true
            }
        } catch {
            throw error
        }
    }
    
    /// Stop audio processing
    func stopAudioProcessing() {
        audioEngine?.stopCapture()
        Task { @MainActor in
            self.isProcessingAudio = false
        }
    }
    
    /// Initialize the neural engine
    private func initializeNeuralEngine() {
        // Only initialize if not already present
        guard neuralEngine == nil else { return }
        
        Task {
            do {
                let engine = try NeuralEngine()
                Task { @MainActor in
                    self.neuralEngine = engine
                }
            } catch {
                handleError("Failed to initialize neural engine: \(error.localizedDescription)")
            }
        }
    }
    
    /// Update visualization parameters
    private func updateVisualizationParameters() {
        visualizer?.setVisualizationParameters(
            sensitivity: sensitivity,
            motionIntensity: motionIntensity,
            colorIntensity: colorIntensity
        )
    }
    
    /// Load settings from app storage
    private func loadSettings() async {
        Task { @MainActor in
            // Load visualization settings
            self.selectedVisualizationMode = VisualizationMode(rawValue: Float(self.settings.visualizationMode)) ?? .spectrum
            self.selectedTheme = Theme(rawValue: Float(self.settings.themeIndex)) ?? .classic
            self.sensitivity = Float(self.settings.sensitivity)
            self.motionIntensity = Float(self.settings.motionIntensity)
            self.colorIntensity = Float(self.settings.colorIntensity)
            self.neuralProcessingEnabled = self.settings.neuralProcessingEnabled
            
            // Apply settings to visualizer
            self.visualizer?.setVisualizationMode(self.selectedVisualizationMode)
            self.visualizer?.setTheme(self.selectedTheme)
            self.updateVisualizationParameters()
        }
    }
    
    /// Save current settings
    func saveSettings() {
        settings.visualizationMode = Int(selectedVisualizationMode.rawValue)
        settings.themeIndex = Int(selectedTheme.rawValue)
        settings.sensitivity = Double(sensitivity)
        settings.motionIntensity = Double(motionIntensity)
        settings.colorIntensity = Double(colorIntensity)
        settings.neuralProcessingEnabled = neuralProcessingEnabled
        settings.save()
    }
    
    /// Request microphone access
    private func requestMicrophoneAccess() async -> Bool {
        #if os(macOS)
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        
        if status == .notDetermined {
            return await AVCaptureDevice.requestAccess(for: .audio)
        }
        
        return status == .authorized
        #else
        return await AVAudioSession.sharedInstance().recordPermission == .granted
        #endif
    }
    
    /// Handle errors
    private func handleError(_ message: String) async {
        Task { @MainActor in
            self.errorMessage = message
            self.showError = true
        }
    }
    
    deinit {
        // Clean up resources
        stopAudioProcessing()
        cancellables.forEach { $0.cancel() }
    }
}

/// Main Content View for the application
@available(macOS 15.0, *)
struct ContentView: Sendable: View {
    @EnvironmentObject private var appState: AppState
    
    var body: some View {
        ZStack {
            // Main visualization view
            if let visualizer = appState.visualizer {
                visualizer.createView()
                    .ignoresSafeArea()
            } else {
                // Loading view
                VStack {
                    ProgressView()
                        .scaleEffect(2.0)
                    Text("Initializing audio visualization...")
                        .padding()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
            }
            
            // Controls overlay (conditionally visible)
            VStack {
                Spacer()
                
                // Controls panel at the bottom
                ControlPanelView()
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(15)
                    .padding()
            }
        }
        .background(Color.black)
        .alert("Error", isPresented: $appState.showError) {
            Button("OK") {
                appState.showError = false
            }
        } message: {
            Text(appState.errorMessage ?? "An unknown error occurred")
        }
        .sheet(isPresented: $appState.showAbout) {
            AboutView()
        }
        .sheet(isPresented: $appState.showAudioSettings) {
            AudioSettingsView()
        }
        .sheet(isPresented: $appState.showVisualizerSettings) {
            VisualizerSettingsView()
        }
    }
}

/// Control panel for visualization settings
@available(macOS 15.0, *)
struct ControlPanelView: Sendable: View {
    @EnvironmentObject private var appState: AppState
    @State private var showControls = false
    
    var body: some View {
        VStack(spacing: 16) {
            // Toggle button to show/hide controls
            Button {
                withAnimation {
                    showControls.toggle()
                }
            } label: {
                Image(systemName: showControls ? "chevron.down" : "chevron.up")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 40, height: 40)
                    .background(Color.gray.opacity(0.3))
                    .clipShape(Circle())
            }
            .buttonStyle(PlainButtonStyle())
            
            if showControls {
                HStack(spacing: 20) {
                    // Visualization mode selector
                    VStack {
                        Text("Mode")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Picker("", selection: $appState.selectedVisualizationMode) {
                            Text("Spectrum").tag(VisualizationMode.spectrum)
                            Text("Waveform").tag(VisualizationMode.waveform)
                            Text("Particles").tag(VisualizationMode.particles)
                            Text("Neural").tag(VisualizationMode.neural)
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .frame(width: 300)
                    }
                    
                    // Theme selector
                    VStack {
                        Text("Theme")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Picker("", selection: $appState.selectedTheme) {
                            Text("Classic").tag(Theme.classic)
                            Text("Neon").tag(Theme.neon)
                            Text("Monochrome").tag(Theme.monochrome)
                            Text("Cosmic").tag(Theme.cosmic)
                            Text("Sunset").tag(Theme.sunset)
                            Text("Ocean").tag(Theme.ocean)
                            Text("Forest").tag(Theme.forest)
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(width: 150)
                    }
                    
                    // Toggle for audio processing
                    VStack {
                        Text("Audio")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Button {
                            if appState.isProcessingAudio {
                                appState.stopAudioProcessing()
                            } else {
                                Task {
                                    try await appState.startAudioProcessing()
                                }
                            }
                        } label: {
                            Image(systemName: appState.isProcessingAudio ? "pause.fill" : "play.fill")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                                .frame(width: 40, height: 30)
                                .background(appState.isProcessingAudio ? Color.red.opacity(0.7) : Color.green.opacity(0.7))
                                .cornerRadius(8)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                
                HStack(spacing: 20) {
                    // Sensitivity slider
                    VStack {
                        Text("Sensitivity: \(appState.sensitivity, specifier: "%.2f")")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Slider(value: $appState.sensitivity, in: 0.1...1.5)
                            .frame(width: 150)
                    }
                    
                    // Motion intensity slider
                    VStack {
                        Text("Motion: \(appState.motionIntensity, specifier: "%.2f")")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Slider(value: $appState.motionIntensity, in: 0.1...1.5)
                            .frame(width: 150)
                    }
                    
                    // Color intensity slider
                    VStack {
                        Text("Color: \(appState.colorIntensity, specifier: "%.2f")")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Slider(value: $appState.colorIntensity, in: 0.1...1.5)
                            .frame(width: 150

import SwiftUI
import AudioProcessor
import MLEngine
import Visualizer
import AudioBloomCore
import Combine

@main
@available(macOS 15.0, *)
struct AudioBloomApp: Sendable: App {
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
@available(macOS 15.0, *)
@MainActor
class PerformanceMonitor: MLProcessingDelegate  {
    // Added @MainActor for UI thread safety
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
