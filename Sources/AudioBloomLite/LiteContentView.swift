import SwiftUI
import Combine

struct LiteContentView: View {
    @ObservedObject var audio: SimpleAudioCapture
    @State private var cancellables = Set<AnyCancellable>()
    @State private var spectrum: [Float] = Array(repeating: 0, count: 512)
    @State private var levels: (Float, Float) = (0, 0)

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 16) {
                Text("AudioBloom Lite")
                    .foregroundColor(.white)
                    .font(.title2)
                HStack(spacing: 1) {
                    ForEach(0..<min(128, spectrum.count), id: \.self) { i in
                        Rectangle()
                            .fill(gradient(for: spectrum[i]))
                            .frame(width: 3, height: CGFloat(max(2.0, spectrum[i] * 180)))
                    }
                }
                .frame(height: 200)
                .onReceive(audio.frequencyPublisher) { data in
                    self.spectrum = data
                }

                HStack(spacing: 8) {
                    LevelBar(value: CGFloat(levels.0))
                    LevelBar(value: CGFloat(levels.1))
                }
                .frame(height: 60)
                .onReceive(audio.levelsPublisher) { lv in
                    self.levels = (lv.left, lv.right)
                }
            }
            .padding()
        }
    }

    private func gradient(for v: Float) -> LinearGradient {
        let t = Double(v)
        return LinearGradient(colors: [
            Color(hue: 0.6, saturation: 1, brightness: 0.4 + t * 0.6),
            Color(hue: 0.85, saturation: 1, brightness: 0.4 + t * 0.6)
        ], startPoint: .bottom, endPoint: .top)
    }
}

private struct LevelBar: View {
    let value: CGFloat
    var body: some View {
        ZStack(alignment: .bottom) {
            RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.08))
            RoundedRectangle(cornerRadius: 6).fill(Color.green)
                .frame(height: max(6, value * 60))
        }
        .frame(width: 20)
    }
}