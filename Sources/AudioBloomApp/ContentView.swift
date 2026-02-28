import SwiftUI
import Visualizer
import Combine

struct ContentView: View {
@EnvironmentObject var renderer: MetalRenderer
@EnvironmentObject var settings: AppSettings
    @EnvironmentObject var uiState: UIState

    @State private var isPlaying = false
    @State private var showSettings = false

    var body: some View {
        ZStack {
            VisualizerContainerView()
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                ControlPanelView(isPlaying: $isPlaying, showSettings: $showSettings)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.black.opacity(0.5))
                            .blur(radius: 3)
                    )
                    .padding()
            }

            SettingsPanelView(isVisible: $showSettings)
                .frame(width: 300)
                .background(Color.black.opacity(0.8))
                .offset(x: showSettings ? 0 : 300)
                .animation(.spring(), value: showSettings)
        }
        .onAppear { isPlaying = true }
        .onDisappear { isPlaying = false }
    }
}

struct VisualizerContainerView: View {
    @EnvironmentObject var renderer: MetalRenderer

    var body: some View {
        ZStack {
            if !renderer.isReady {
                Color.black
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: Color.white))
                    .scaleEffect(2.0)
            }
            MetalView(renderer: renderer)
        }
    }
}

struct ControlPanelView: View {
    @Binding var isPlaying: Bool
    @Binding var showSettings: Bool
    @EnvironmentObject var uiState: UIState

    var body: some View {
        HStack(spacing: 20) {
            Button(action: {
                isPlaying.toggle()
                // playback toggle is handled by app; no action needed here
            }) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 36))
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)

            Spacer()
            AudioLevelsView()
            Spacer()

            Button(action: { showSettings.toggle() }) {
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

struct SettingsPanelView: View {
    @Binding var isVisible: Bool
    @EnvironmentObject var settings: AudioBloomSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Settings").font(.title2).fontWeight(.bold)
                Spacer()
                Button(action: { isVisible = false }) {
                    Image(systemName: "xmark.circle.fill").font(.title3)
                }
            }
            .padding(.bottom)

            Group {
                VStack(alignment: .leading) {
                    Text("Theme").fontWeight(.medium)
Picker("Theme", selection: $settings.currentTheme) {
                        ForEach([Theme.classic, .neon, .monochrome, .cosmic, .sunset, .ocean, .forest], id: \.self) { theme in
                            Text(String(describing: theme)).tag(theme)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }

                Toggle("Enable Neural Engine", isOn: $settings.neuralEngineEnabled)

                VStack(alignment: .leading) {
                    Text("Target Frame Rate: \(settings.frameRateTarget)").fontWeight(.medium)
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

struct AudioLevelBar: View {
    var level: CGFloat
    var body: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(levelColor)
            .frame(width: 6, height: 20 * level)
            .frame(height: 20, alignment: .bottom)
    }
    private var levelColor: Color {
        if level < 0.3 { return .green }
        else if level < 0.7 { return .yellow }
        else { return .red }
    }
}

struct AudioLevelsView: View {
    @EnvironmentObject var audioEngine: AudioEngine
    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<10, id: \.self) { index in
                AudioLevelBar(level: barLevel(at: index))
            }
        }
        .frame(width: 100, height: 30)
    }
    private func barLevel(at index: Int) -> CGFloat {
        let avgLevel = (uiState.levels.left + uiState.levels.right) / 2.0
        let threshold = Float(index) * 0.1 + 0.05
        if avgLevel > threshold { return CGFloat(min(1.0, (avgLevel - threshold) * 2.0)) } else { return 0.0 }
    }
}
