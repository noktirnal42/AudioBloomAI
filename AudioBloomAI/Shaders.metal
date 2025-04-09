#include <metal_stdlib>
using namespace metal;

// Data structures
struct VertexIn {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float4 color;
};

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 projectionMatrix;
    float time;
    float intensity;
    float visualizationType; // 0: waveform, 1: spectrum, 2: circular
};

struct AudioData {
    device float* samples;     // Raw audio samples for waveform
    device float* fftData;     // FFT data for spectrum
    uint sampleCount;          // Number of samples
    uint fftDataSize;          // Size of FFT data
    float amplitude;           // Global amplitude multiplier
};

// Color utilities
float4 hsv2rgb(float h, float s, float v) {
    float3 rgb = clamp(abs(fmod(h * 6.0 + float3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = rgb * rgb * (3.0 - 2.0 * rgb); // cubic smoothing
    return float4(v * mix(float3(1.0), rgb, s), 1.0);
}

float4 getGradientColor(float position, float intensity) {
    // Create a gradient based on position (0-1) and audio intensity
    float hue = fmod(position * 0.8 + intensity * 0.2, 1.0);
    float saturation = 0.7 + intensity * 0.3;
    float value = 0.6 + intensity * 0.4;
    
    return hsv2rgb(hue, saturation, value);
}

// ====== WAVEFORM VISUALIZATION ======

// Vertex shader for waveform visualization
vertex VertexOut waveformVertexShader(const VertexIn vertexIn [[stage_in]],
                                     constant Uniforms &uniforms [[buffer(1)]],
                                     constant AudioData &audioData [[buffer(2)]],
                                     uint vertexID [[vertex_id]]) {
    VertexOut vertexOut;
    
    // Calculate vertex position based on audio sample
    float x = float(vertexID) / float(audioData.sampleCount - 1) * 2.0 - 1.0; // Range: -1 to 1
    
    // Get audio sample value and scale it
    float sampleValue = 0.0;
    if (vertexID < audioData.sampleCount) {
        sampleValue = audioData.samples[vertexID] * audioData.amplitude;
    }
    sampleValue = clamp(sampleValue, -0.95, 0.95); // Prevent going out of bounds
    
    // Apply animations based on time
    float animatedSample = sampleValue * (1.0 + sin(uniforms.time * 2.0) * 0.1);
    
    // Set vertex position
    vertexOut.position = uniforms.projectionMatrix * uniforms.modelMatrix * float4(x, animatedSample, 0.0, 1.0);
    vertexOut.texCoord = vertexIn.texCoord;
    
    // Generate color based on position and amplitude
    float intensity = abs(sampleValue) * 2.0;
    vertexOut.color = getGradientColor(float(vertexID) / float(audioData.sampleCount), intensity);
    
    return vertexOut;
}

// Fragment shader for waveform visualization
fragment float4 waveformFragmentShader(VertexOut fragmentIn [[stage_in]],
                                      constant Uniforms &uniforms [[buffer(1)]]) {
    // Apply subtle glow effect based on position
    float4 baseColor = fragmentIn.color;
    float glow = pow(1.0 - abs(fragmentIn.texCoord.y), 5.0) * 0.5;
    
    // Brighten color near center of line
    return baseColor + float4(glow, glow, glow, 0.0);
}

// ====== SPECTRUM VISUALIZATION ======

// Vertex shader for spectrum visualization (FFT)
vertex VertexOut spectrumVertexShader(const VertexIn vertexIn [[stage_in]],
                                      constant Uniforms &uniforms [[buffer(1)]],
                                      constant AudioData &audioData [[buffer(2)]],
                                      uint vertexID [[vertex_id]]) {
    VertexOut vertexOut;
    
    // Calculate bar index and whether this is the top or bottom vertex of the bar
    uint barIndex = vertexID / 2;
    bool isTop = (vertexID % 2) == 0;
    
    // Ensure we don't go out of bounds
    barIndex = min(barIndex, audioData.fftDataSize - 1);
    
    // Calculate x position (normalized from -1 to 1)
    float x = (float(barIndex) / float(audioData.fftDataSize - 1)) * 2.0 - 1.0;
    
    // Get FFT magnitude value for this bar
    float magnitude = audioData.fftData[barIndex] * audioData.amplitude;
    magnitude = clamp(magnitude, 0.0, 1.0);
    
    // Add some animation to the bars
    float animatedMagnitude = magnitude * (1.0 + sin(uniforms.time * 3.0 + x * 2.0) * 0.1);
    
    // Set y position based on whether this is top or bottom vertex
    float y = isTop ? animatedMagnitude : -0.05;
    
    // Set vertex position
    vertexOut.position = uniforms.projectionMatrix * uniforms.modelMatrix * float4(x, y, 0.0, 1.0);
    vertexOut.texCoord = vertexIn.texCoord;
    
    // Generate color based on frequency (bar position) and magnitude
    float intensity = magnitude;
    vertexOut.color = getGradientColor(float(barIndex) / float(audioData.fftDataSize), intensity);
    
    return vertexOut;
}

// Fragment shader for spectrum visualization
fragment float4 spectrumFragmentShader(VertexOut fragmentIn [[stage_in]],
                                       constant Uniforms &uniforms [[buffer(1)]]) {
    // Apply vertical gradient to each bar
    float verticalGradient = 1.0 - fragmentIn.texCoord.y;
    float4 color = fragmentIn.color * (0.7 + verticalGradient * 0.3);
    
    // Add pulsing glow effect
    float pulse = (sin(uniforms.time * 2.0) + 1.0) * 0.15;
    return color + float4(pulse, pulse, pulse, 0.0);
}

// ====== CIRCULAR VISUALIZATION ======

// Vertex shader for circular visualization
vertex VertexOut circularVertexShader(const VertexIn vertexIn [[stage_in]],
                                      constant Uniforms &uniforms [[buffer(1)]],
                                      constant AudioData &audioData [[buffer(2)]],
                                      uint vertexID [[vertex_id]]) {
    VertexOut vertexOut;
    
    // Calculate angle around the circle
    float angle = float(vertexID) / float(audioData.sampleCount) * M_PI_F * 2.0;
    
    // Get audio sample value
    float sampleValue = 0.0;
    if (vertexID < audioData.sampleCount) {
        sampleValue = audioData.samples[vertexID] * audioData.amplitude;
    }
    sampleValue = clamp(sampleValue, -0.95, 0.95);
    
    // Base radius plus sample offset
    float radius = 0.5 + sampleValue * 0.4;
    
    // Add ripple effect based on time
    radius += sin(uniforms.time * 1.5 + angle * 4.0) * 0.05;
    
    // Calculate position using polar coordinates
    float x = cos(angle) * radius;
    float y = sin(angle) * radius;
    
    // Set vertex position
    vertexOut.position = uniforms.projectionMatrix * uniforms.modelMatrix * float4(x, y, 0.0, 1.0);
    vertexOut.texCoord = vertexIn.texCoord;
    
    // Generate color based on angle and amplitude
    float intensity = (radius - 0.5) * 2.0 + 0.5;
    float hueOffset = fmod(uniforms.time * 0.1, 1.0);
    vertexOut.color = getGradientColor(fmod(angle / (M_PI_F * 2.0) + hueOffset, 1.0), intensity);
    
    return vertexOut;
}

// Fragment shader for circular visualization
fragment float4 circularFragmentShader(VertexOut fragmentIn [[stage_in]],
                                       constant Uniforms &uniforms [[buffer(1)]]) {
    // Apply radial glow
    float4 baseColor = fragmentIn.color;
    float4 finalColor = baseColor;
    
    // Add subtle pulse effect
    float pulse = sin(uniforms.time) * 0.1 + 0.9;
    finalColor *= pulse;
    
    return finalColor;
}

// ====== DEFAULT FALLBACK SHADERS ======

// Basic vertex shader (fallback)
vertex VertexOut vertexShader(const VertexIn vertexIn [[stage_in]],
                             constant Uniforms &uniforms [[buffer(1)]]) {
    VertexOut vertexOut;
    vertexOut.position = uniforms.projectionMatrix * uniforms.modelMatrix * float4(vertexIn.position, 1.0);
    vertexOut.texCoord = vertexIn.texCoord;
    vertexOut.color = float4(1.0, 1.0, 1.0, 1.0);
    return vertexOut;
}

// Basic fragment shader (fallback)
fragment float4 fragmentShader(VertexOut fragmentIn [[stage_in]]) {
    return fragmentIn.color;
}

