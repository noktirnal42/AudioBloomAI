import SwiftUI
import AudioBloomCore
import CoreAudio
import Visualizer
import MLEngine

/// Main content view for the application
struct ContentView: View {
    /// Application settings
    @EnvironmentObject var settings: AudioBloomSettings
    
    /// Reference to the audio engine
    @EnvironmentObject var audioEngine: AudioEngine
    
    /// Reference to the Metal renderer
    @EnvironmentObject var renderer: MetalRenderer
    
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
struct VisualizerContainerView: View {
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
struct ControlPanelView: View {
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

/// Audio levels visualization
struct AudioLevelsView: View {
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
        let avgLevel = (audioEngine.levels.left + audioEngine.levels.right)

