// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
import SwiftUI
import AudioProcessor
import MLEngine
import Visualizer
import AudioBloomCore
import Combine

@available(macOS 15.0, *)
struct ContentView: Sendable: View {
    // Access environment objects
    @EnvironmentObject var audioEngine: AudioEngine
    @EnvironmentObject var audioBridge: AudioBridge
    
    // State for visualization data
    @State private var visualizationData: VisualizationData?
    @State private var frequencyData: [Float] = []
    @State private var audioLevels: (left: Float, right: Float) = (0, 0)
    
    // Subscriptions
    @State private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("AudioBloom AI")
                .font(.largeTitle)
                .padding()
            
            // Audio levels display
            HStack(spacing: 20) {
                LevelMeterView(level: audioLevels.left)
                    .frame(width: 30, height: 200)
                
                // Visualization area
                ZStack {
                    // Primary visualization
                    FrequencyVisualizationView(data: visualizationData?.values ?? frequencyData)
                        .frame(height: 200)
                        .background(Color.black.opacity(0.1))
                        .cornerRadius(10)
                    
                    // Beat indicator (if significant event detected)
                    if visualizationData?.isSignificantEvent == true {
                        Circle()
                            .fill(Color.red.opacity(0.7))
                            .frame(width: 20, height: 20)
                            .position(x: 20, y: 20)
                    }
                }
                
                LevelMeterView(level: audioLevels.right)
    
    /// Reference to the ML processor
    @EnvironmentObject var mlProcessor: MLProcessor
    
    /// Visualization state
    @State private var isPlaying = false
    
    /// Show settings panel
    @State private var showSettings = false
    
    var body: some View {
        ZStack {
            // Main visualization view
            VisualizerContainerView()
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                
                // Control panel
                ControlPanelView(isPlaying: $isPlaying, showSettings: $showSettings)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.black.opacity(0.5))
                            .blur(radius: 3)
                    )
                    .padding()
            }
            
            // Settings panel (slides in from right when active)
            SettingsPanelView(isVisible: $showSettings)
                .frame(width: 300)
                .background(Color.black.opacity(0.8))
                .offset(x: showSettings ? 0 : 300)
                .animation(.spring(), value: showSettings)
        }
        .onAppear {
            // Start audio capture when view appears
            try? audioEngine.startCapture()
            isPlaying = true
        }
        .onDisappear {
            // Stop audio capture when view disappears
            audioEngine.stopCapture()
            isPlaying = false
        }
    }
}

/// Container for the Metal visualizer
@available(macOS 15.0, *)
struct VisualizerContainerView: Sendable: View {
    /// Reference to the Metal renderer
    @EnvironmentObject var renderer: MetalRenderer
    
    var body: some View {
        ZStack {
            // Placeholder for when renderer is not ready
            if !renderer.isReady {
                Color.black
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: Color.white))
                    .scaleEffect(2.0)
            }
            
            // Metal view for rendering
            MetalView(renderer: renderer)
        }
    }
}

/// Control panel for audio visualization
@available(macOS 15.0, *)
struct ControlPanelView: Sendable: View {
    /// Audio playback state
    @Binding var isPlaying: Bool
    
    /// Settings panel visibility
    @Binding var showSettings: Bool
    
    /// Reference to the audio engine
    @EnvironmentObject var audioEngine: AudioEngine
    
    var body: some View {
        HStack(spacing: 20) {
            // Play/Pause button
            Button(action: {
                isPlaying.toggle()
                if isPlaying {
                    try? audioEngine.startCapture()
                } else {
                    audioEngine.stopCapture()
                }
            }) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 36))
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            // Audio levels visualization
            AudioLevelsView()
            
            Spacer()
            
            // Settings button
            Button(action: {
                showSettings.toggle()
            }) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 24))
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)
        }
        .frame(height: 60)
        .padding(.horizontal)
    }
}

/// Settings panel view for adjusting visualization parameters
@available(macOS 15.0, *)
struct SettingsPanelView: Sendable: View {
    /// Binding for visibility control
    @Binding var isVisible: Bool
    
    /// Application settings
    @EnvironmentObject var settings: AudioBloomSettings
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Settings")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: {
                    isVisible = false
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                }
            }
            .padding(.bottom)
            
            Group {
                VStack(alignment: .leading) {
                    Text("Theme")
                        .fontWeight(.medium)
                    
                    Picker("Theme", selection: $settings.currentTheme) {
                        ForEach(AudioBloomCore.VisualTheme.allCases) { theme in
                            Text(theme.rawValue).tag(theme)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                Toggle("Enable Neural Engine", isOn: $settings.neuralEngineEnabled)
                
                VStack(alignment: .leading) {
                    Text("Target Frame Rate: \(settings.frameRateTarget)")
                        .fontWeight(.medium)
                    
                    Slider(value: Binding(
                        get: { Double(settings.frameRateTarget) },
                        set: { settings.frameRateTarget = Int($0) }
                    ), in: 30...120, step: 10)
                }
            }
            
            Spacer()
        }
        .padding()
        .frame(maxHeight: .infinity)
        .foregroundColor(.white)
    }
}

/// Audio level bar visualization component
@available(macOS 15.0, *)
struct AudioLevelBar: Sendable: View {
    /// The current level (0-1)
    var level: CGFloat
    
    var body: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(levelColor)
            .frame(width: 6, height: 20 * level)
            .frame(height: 20, alignment: .bottom)
    }
    
    /// Color based on level
    private var levelColor: Color {
        if level < 0.3 {
            return .green
        } else if level < 0.7 {
            return .yellow
        } else {
            return .red
        }
    }
}

/// Audio levels visualization
@available(macOS 15.0, *)
struct AudioLevelsView: Sendable: View {
    /// Reference to the audio engine
    @EnvironmentObject var audioEngine: AudioEngine
    
    var body: some View {
        HStack(spacing: 3) {
            // Display 10 level bars for visual feedback
            ForEach(0..<10, id: \.self) { index in
                AudioLevelBar(level: barLevel(at: index))
            }
        }
        .frame(width: 100, height: 30)
    }
    
    /// Calculate the level for a specific bar
    private func barLevel(at index: Int) -> CGFloat {
        let avgLevel = (audioEngine.levels.left + audioEngine.levels.right) / 2.0
        
        // Create a logarithmic distribution for the bars
        // Lower index bars show lower frequency activity
        let threshold = Float(index) * 0.1 + 0.05
        
        // Check if the level exceeds the threshold for this bar
        if avgLevel > threshold {
            return CGFloat(min(1.0, (avgLevel - threshold) * 2.0))
        } else {
            return 0.0
        }
    }
}
