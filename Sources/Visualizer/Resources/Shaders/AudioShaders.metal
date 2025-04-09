//
// AudioShaders.metal
// AudioBloomAI
//
// Metal shaders for audio visualization effects
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Structures

// Input from vertex buffer
struct VertexIn {
    float4 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

// Output to fragment shader
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float4 color;
    float4 userData;         // Additional data to pass to fragment shader
};

// Uniforms structure matching the renderer implementation
struct AudioUniforms {
    // Audio data
    float audioData[1024];       // FFT frequency data
    float bassLevel;             // Low frequency intensity
    float midLevel;              // Mid frequency intensity
    float trebleLevel;           // High frequency intensity
    float leftLevel;             // Left channel volume
    float rightLevel;            // Right channel volume
    
    // Theme colors
    float4 primaryColor;         // Primary theme color
    float4 secondaryColor;       // Secondary theme color
    float4 backgroundColor;      // Background color
    float4 accentColor;          // Accent color for highlights
    
    // Animation parameters
    float time;                  // Current time in seconds
    float sensitivity;           // Audio sensitivity (0.0-1.0)
    float motionIntensity;       // Motion intensity (0.0-1.0)
    float themeIndex;            // Current theme index (0-7 for different themes)
    
    // Visualization settings
    float visualizationMode;     // 0: Spectrum, 1: Waveform, 2: Particles, 3: Neural
    float previousMode;          // Previous visualization mode for transitions
    float transitionProgress;    // Progress between mode transitions (0.0-1.0)
    float colorIntensity;        // Color intensity parameter (0.0-1.0)
    
    // Processing parameters
    float spectrumSmoothing;     // Spectrum smoothing factor (0.0-1.0)
    float particleCount;         // Desired particle count (translated to density)
    
    // Neural visualization parameters
    float neuralEnergy;          // Energy level from neural analysis
    float neuralPleasantness;    // Pleasantness factor from neural analysis
    float neuralComplexity;      // Complexity factor from neural analysis
    float beatDetected;          // Beat detection flag (0.0 or 1.0)
};

// MARK: - Helper Functions

// Convert HSV to RGB color space
float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Get audio sample with smoothing
float getAudioSample(constant AudioUniforms &uniforms, int index, int maxIndex) {
    // Apply bounds checking
    index = clamp(index, 0, maxIndex);
    
    // Apply smoothing if enabled
    if (uniforms.spectrumSmoothing > 0.01) {
        float value = uniforms.audioData[index];
        float smoothValue = 0.0;
        int smoothWidth = max(1, int(uniforms.spectrumSmoothing * 10.0));
        
        // Average neighboring values
        for (int i = -smoothWidth; i <= smoothWidth; i++) {
            int sampleIndex = clamp(index + i, 0, maxIndex);
            float weight = 1.0 - abs(float(i)) / float(smoothWidth + 1);
            smoothValue += uniforms.audioData[sampleIndex] * weight;
        }
        
        // Normalize
        smoothValue /= float(smoothWidth * 2 + 1);
        
        // Mix between raw and smoothed based on smoothing factor
        return mix(value, smoothValue, uniforms.spectrumSmoothing) * uniforms.sensitivity;
    } else {
        return uniforms.audioData[index] * uniforms.sensitivity;
    }
}

// Random number generation
float random(float2 p) {
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

// 2D Noise function
float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float a = random(i);
    float b = random(i + float2(1.0, 0.0));
    float c = random(i + float2(0.0, 1.0));
    float d = random(i + float2(1.0, 1.0));
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// FBM (Fractal Brownian Motion) for more natural noise
float fbm(float2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for(int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// MARK: - Visualization Mode Distortion Functions

// Function to compute vertex distortion for Spectrum visualization
float2 spectrum_vertex_distortion(float2 texCoord, constant AudioUniforms &uniforms) {
    float xOffset = texCoord.x * 64.0;
    int audioIndex = int(xOffset) % 1024;
    float audioValue = getAudioSample(uniforms, audioIndex, 1023);
    float bassEffect = uniforms.bassLevel * uniforms.motionIntensity * 0.3;
    float timeScale = uniforms.time * 0.5;
    
    // Bar-like vertical distortion
    float barHeight = audioValue * uniforms.sensitivity * uniforms.motionIntensity;
    float yDistortion = step(1.0 - barHeight, texCoord.y) * barHeight * 0.1;
    
    // Horizontal wave effect
    float xDistortion = sin(texCoord.y * 10.0 + timeScale) * 0.02 * audioValue;
    
    return float2(xDistortion, yDistortion);
}

// Function to compute vertex distortion for Waveform visualization
float2 waveform_vertex_distortion(float2 texCoord, constant AudioUniforms &uniforms) {
    float timeScale = uniforms.time * 0.3;
    float yPos = texCoord.y * 2.0 - 1.0; // Center at 0
    
    // Get audio value at position
    int audioIndex = int(texCoord.x * 1024.0) % 1024;
    float audioValue = uniforms.audioData[audioIndex] * uniforms.sensitivity;
    
    // Create wave effect centered at y=0.5
    float waveOffset = (sin(texCoord.x * 30.0 + timeScale) * 0.2 + audioValue) * uniforms.motionIntensity;
    float distFromCenter = abs(texCoord.y - 0.5) * 2.0;
    float waveAttenuated = waveOffset * (1.0 - distFromCenter);
    
    return float2(0.0, waveAttenuated);
}

// Function to compute vertex distortion for Particles visualization
float2 particles_vertex_distortion(float2 texCoord, constant AudioUniforms &uniforms) {
    float timeScale = uniforms.time * 0.4;
    float noiseValue = fbm(texCoord * 3.0 + timeScale * 0.2, 3);
    float audioReactivity = mix(uniforms.bassLevel, uniforms.trebleLevel, texCoord.y);
    
    // Create swirling motion with noise
    float xDistortion = sin(texCoord.y * 5.0 + noiseValue * 10.0 + timeScale) * 0.03;
    float yDistortion = cos(texCoord.x * 5.0 + noiseValue * 10.0 + timeScale) * 0.03;
    
    // Amplify with audio and motion intensity
    xDistortion *= uniforms.motionIntensity * (1.0 + audioReactivity);
    yDistortion *= uniforms.motionIntensity * (1.0 + audioReactivity);
    
    return float2(xDistortion, yDistortion);
}

// Function to compute vertex distortion for Neural visualization
float2 neural_vertex_distortion(float2 texCoord, constant AudioUniforms &uniforms) {
    float timeScale = uniforms.time * 0.3;
    float neuralFactor = uniforms.neuralEnergy * uniforms.neuralComplexity;
    float beatPulse = uniforms.beatDetected * sin(uniforms.time * 10.0) * 0.05;
    
    // Create organic flowing motion based on neural parameters
    float complexity = mix(3.0, 8.0, uniforms.neuralComplexity);
    float flowSpeed = mix(0.2, 0.8, uniforms.neuralPleasantness);
    
    // Create complex, organic distortion
    float noiseValue = fbm(texCoord * complexity + timeScale * flowSpeed, 4);
    float xDistortion = sin(noiseValue * 10.0 + timeScale) * 0.04 * neuralFactor;
    float yDistortion = cos(noiseValue * 8.0 + timeScale * 0.7) * 0.04 * neuralFactor;
    
    // Add beat reaction
    xDistortion += beatPulse * (texCoord.x - 0.5);
    yDistortion += beatPulse * (texCoord.y - 0.5);
    
    return float2(xDistortion, yDistortion) * uniforms.motionIntensity;
}

// MARK: - Fragment Visualization Effects

// Spectrum visualization fragment effects
float4 spectrum_fragment_effect(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, float audioIntensity) {
    float4 color = baseColor;
    
    // Get frequency data
    int freqIndex = int(uv.x * 64);
    float freqValue = getAudioSample(uniforms, freqIndex, 1023);
    
    // Create vertical bars
    float barWidth = 0.02 + uniforms.sensitivity * 0.01;
    float barX = fract(uv.x / barWidth);
    float barIntensity = step(barX, 0.8); // Bar width (80% of spacing)
    
    // Bar height based on audio data
    float barHeight = freqValue * uniforms.sensitivity * 1.5;
    float barY = step(1.0 - barHeight, uv.y);
    
    // Bar color intensity
    float intensity = barIntensity * barY * (1.0 + uniforms.bassLevel * 0.5);
    
    // Apply bar coloring - spectrum gradient
    float freqHue = uv.x; // Hue based on frequency
    float3 barColor = hsv2rgb(float3(freqHue, 0.8, 1.0));
    
    // Apply different themes
    if (uniforms.themeIndex < 1.0) {
        // Classic theme - rainbow spectrum
        color.rgb = mix(color.rgb, barColor, intensity * 0.8);
    } else if (uniforms.themeIndex < 2.0) {
        // Neon theme - bright with glow
        color.rgb = mix(color.rgb, uniforms.primaryColor.rgb, intensity * 0.7);
        color.rgb += uniforms.secondaryColor.rgb * intensity * 0.5 * (sin(uniforms.time * 3.0) * 0.3 + 0.7);
    } else if (uniforms.themeIndex < 3.0) {
        // Monochrome theme - single color bars
        color.rgb = mix(color.rgb, float3(1.0), intensity * 0.9);
    } else {
        // Cosmic theme - starfield with spectrum
        float3 cosmicColor = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, sin(uv.y * 10.0 + uniforms.time) * 0.5 + 0.5);
        color.rgb = mix(color.rgb, cosmicColor, intensity * 0.7);
        
        // Add stars
        float starNoise = random(uv * 50.0);
        float star = step(0.98, starNoise) * (sin(uniforms.time * 3.0 + starNoise * 10.0) * 0.5 + 0.5);
        color.rgb += star * uniforms.accentColor.rgb;
    }
    
    return color;
}

// Waveform visualization fragment effects
float4 waveform_fragment_effect(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, float audioIntensity) {
    float4 color = baseColor;
    
    // Create oscilloscope-like effect
    float centerY = 0.5; // Center of waveform
    float waveThickness = 0.02 * uniforms.sensitivity;
    
    // Calculate waveform height at this x-position
    int waveIndex = int(uv.x * 1024);
    float waveHeight = uniforms.audioData[waveIndex] * uniforms.sensitivity;
    float waveY = centerY + waveHeight * 0.4; // Scale for visibility
    
    // Calculate distance to the wave line
    float dist = abs(uv.y - waveY);
    float waveIntensity = smoothstep(waveThickness, 0.0, dist);
    
    // Add time-based movement
    float timeOffset = sin(uniforms.time * 2.0) * 0.01 * uniforms.motionIntensity;
    waveIntensity += smoothstep(waveThickness, 0.0, abs(uv.y - (waveY + timeOffset))) * 0.5;
    
    // Apply different themes
    if (uniforms.themeIndex < 1.0) {
        // Classic theme - crisp line
        color.rgb = mix(color.rgb, uniforms.primaryColor.rgb, waveIntensity);
        
        // Add subtle background grid
        float grid = max(
            smoothstep(0.03, 0.02, abs(fract(uv.x * 10.0) - 0.5)),
            smoothstep(0.03, 0.02, abs(fract(uv.y * 10.0) - 0.5))
        ) * 0.2;
        color.rgb = mix(color.rgb, uniforms.secondaryColor.rgb, grid);
    } else if (uniforms

