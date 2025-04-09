// AudioVisualizer.metal
// Complete shader implementation for AudioBloomAI audio visualizer
//

#include <metal_stdlib>
using namespace metal;

// Structures for passing data between shaders
struct VertexIn {
    float4 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float4 color;
    float4 userData;         // Additional data to pass to fragment shader
};

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
    float transitionProgress;    // Progress between mode transitions (0.0-1.0)
    float previousMode;          // Previous visualization mode for transitions
    float colorIntensity;        // Color intensity parameter (0.0-1.0)
    float spectrumSmoothing;     // Spectrum smoothing factor (0.0-1.0)
    float particleCount;         // Desired particle count (translated to density)
    
    // Neural visualization parameters (when using neural network outputs)
    float neuralEnergy;          // Overall energy detected by neural analysis
    float neuralPleasantness;    // Pleasantness factor from neural analysis
    float neuralComplexity;      // Complexity factor from neural analysis
    float beatDetected;          // Beat detection flag (0.0 or 1.0)
};

//-----------------------------------------------------------
// Helper functions for visualization effects
//-----------------------------------------------------------

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

//-----------------------------------------------------------
// Vertex Distortion Functions for Different Visualization Modes
//-----------------------------------------------------------

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

//-----------------------------------------------------------
// Vertex Shader Implementation
//-----------------------------------------------------------

vertex VertexOut audio_vertex_shader(uint vertexID [[vertex_id]],
                                    constant float4 *positions [[buffer(0)]],
                                    constant AudioUniforms &uniforms [[buffer(1)]]) {
    VertexOut out;
    
    // Get base position from vertex buffer
    float4 basePosition = positions[vertexID];
    float2 texCoord = float2((basePosition.x + 1.0) * 0.5, (basePosition.y + 1.0) * 0.5);
    
    // Initialize distortion
    float2 distortion = float2(0.0, 0.0);
    float2 distortion1 = float2(0.0, 0.0);
    float2 distortion2 = float2(0.0, 0.0);
    
    // Get current visualization mode (with bounds checking)
    int currentMode = int(uniforms.visualizationMode);
    currentMode = clamp(currentMode, 0, 3); // Clamp to valid range [0,3]
    
    // Get previous mode for transitions (with bounds checking)
    int prevMode = int(uniforms.previousMode);
    prevMode = clamp(prevMode, 0, 3); // Clamp to valid range [0,3]
    
    // Calculate distortion based on current mode
    if (currentMode == 0) {
        distortion1 = spectrum_vertex_distortion(texCoord, uniforms);
    } else if (currentMode == 1) {
        distortion1 = waveform_vertex_distortion(texCoord, uniforms);
    } else if (currentMode == 2) {
        distortion1 = particles_vertex_distortion(texCoord, uniforms);
    } else if (currentMode == 3) {
        distortion1 = neural_vertex_distortion(texCoord, uniforms);
    }
    
    // If in transition, calculate distortion for previous mode too
    if (uniforms.transitionProgress > 0.0 && uniforms.transitionProgress < 1.0) {
        if (prevMode == 0) {
            distortion2 = spectrum_vertex_distortion(texCoord, uniforms);
        } else if (prevMode == 1) {
            distortion2 = waveform_vertex_distortion(texCoord, uniforms);
        } else if (prevMode == 2) {
            distortion2 = particles_vertex_distortion(texCoord, uniforms);
        } else if (prevMode == 3) {
            distortion2 = neural_vertex_distortion(texCoord, uniforms);
        }
        
        // Smoothly interpolate between distortions based on transition progress
        distortion = mix(distortion2, distortion1, uniforms.transitionProgress);
    } else {
        distortion = distortion1;
    }
    
    // Apply distortion to position
    out.position = basePosition;
    out.position.x += distortion.x;
    out.position.y += distortion.y;
    
    // Calculate color based on position, audio and theme
    float audioIntensity = 0.0;
    
    // Get frequency data based on position for coloring
    int colorAudioIndex = int(texCoord.x * 64.0) % 1024;
    float colorAudioValue = getAudioSample(uniforms, colorAudioIndex, 1023);
    
    // Mix audio values for better coloring
    audioIntensity = mix(
        colorAudioValue,
        mix(uniforms.bassLevel, uniforms.trebleLevel, texCoord.y),
        0.5
    );
    
    // Boost color with audio reactivity
    float brightness = (audioIntensity * uniforms.sensitivity) * 0.5 + 0.5;
    brightness = clamp(brightness * uniforms.colorIntensity, 0.0, 1.0);
    
    // Create color based on theme and position
    float themeHue = fract(texCoord.x + texCoord.y * 0.5 + uniforms.time * 0.05);
    
    // Primary color influences
    float4 primaryInfluence = uniforms.primaryColor * brightness;
    
    // Secondary color influences (based on audio)
    float audioModulation = sin(texCoord.x * 5.0 + uniforms.time) * 0.5 + 0.5;
    float4 secondaryInfluence = uniforms.secondaryColor * audioModulation * audioIntensity;
    
    // Combine colors
    out.color = mix(primaryInfluence, secondaryInfluence, 0.5);
    out.color.a = 1.0;
    
    // Store texture coordinates for fragment shader
    out.texCoord = texCoord;
    
    // Pass additional data to fragment shader
    out.userData = float4(
        audioIntensity,                 // Audio intensity at this position
        float(currentMode),             // Current visualization mode
        uniforms.transitionProgress,    // Transition progress
        uniforms.beatDetected           // Beat detection flag
    );
    
    return out;
}

//-----------------------------------------------------------
// Fragment Shader Functions for Different Visualization Modes
//-----------------------------------------------------------

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
    } else if (uniforms.themeIndex < 2.0) {
        // Neon theme - glowing line
        color.rgb = mix(color.rgb, uniforms.primaryColor.rgb, waveIntensity);
        // Add glow effect
        float glow = smoothstep(waveThickness * 3.0, 0.0, dist) * 0.7;
        color.rgb += uniforms.secondaryColor.rgb * glow * (sin(uniforms.time * 5.0) * 0.2 + 0.8);
    } else if (uniforms.themeIndex < 3.0) {
        // Monochrome theme - simple black & white
        color.rgb = mix(color.rgb, float3(1.0), waveIntensity);
        // Add noise texture
        float staticNoise = random(uv * 50.0 + uniforms.time * 0.1) * 0.1;
        color.rgb += float3(staticNoise);
    } else {
        // Cosmic theme - multi-colored wave
        float waveHue = fract(uv.x + uniforms.time * 0.1);
        float3 waveColor = hsv2rgb(float3(waveHue, 0.7, 1.0));
        color.rgb = mix(color.rgb, waveColor, waveIntensity);
        
        // Add subtle nebula background
        float nebula = fbm(uv * 3.0 + uniforms.time * 0.05, 3) * 0.3;
        color.rgb += nebula * uniforms.accentColor.rgb;
    }
    
    return color;
}

// Particle visualization fragment effects
float4 particles_fragment_effect(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, float audioIntensity) {
    float4 color = baseColor;
    
    // Particle system parameters
    float particleDensity = max(0.1, min(1.0, uniforms.particleCount / 100.0)); // Normalized 0.1-1.0
    float particleSize = 0.005 + uniforms.sensitivity * 0.01;
    float timeScale = uniforms.time * 0.3;
    
    // Generate particle field
    float particles = 0.0;
    
    // Optimize by limiting number of particles based on density
    int particleIterations = max(3, int(particleDensity * 10.0));
    
    for (int i = 0; i < particleIterations; i++) {
        // Create varied particle positions
        float2 particlePos = float2(
            fract(random(float2(float(i) * 0.45, 0.23)) + timeScale * (0.1 + random(float2(float(i), 0.92)) * 0.1)),
            fract(random(float2(float(i) * 0.39, 0.87)) + timeScale * (0.1 + random(float2(0.75, float(i))) * 0.05))
        );
        
        // Add audio reactivity to particle movement
        float freqIndex = i % 64;
        float audioReactivity = getAudioSample(uniforms, freqIndex, 63);
        particlePos.y += sin(timeScale * 5.0 + float(i)) * audioReactivity * 0.2;
        
        // Calculate distance to particle center
        float dist = length(uv - particlePos);
        
        // Apply audio reactivity to particle size
        float dynamicSize = particleSize * (1.0 + audioReactivity * 2.0);
        
        // Smooth particle shape
        float particle = smoothstep(dynamicSize, 0.0, dist);
        
        // Accumulate particle contribution
        particles += particle;
    }
    
    // Clamp to avoid oversaturation with many particles
    particles = min(particles, 1.0);
    
    // Apply different themes
    if (uniforms.themeIndex < 1.0) {
        // Classic theme - colorful particles
        float3 particleColor = hsv2rgb(float3(
            fract(uniforms.time * 0.1),  // Slowly changing hue
            0.7,                         // Saturation
            particles                    // Brightness from particle intensity
        ));
        color.rgb = mix(color.rgb, particleColor, particles * 0.8);
    } else if (uniforms.themeIndex < 2.0) {
        // Neon theme - bright glowing particles
        float3 glow = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                         sin(uniforms.time + uv.x * 5.0) * 0.5 + 0.5);
        color.rgb = mix(color.rgb, glow, particles * 0.8);
        // Add extra glow for neon effect
        color.rgb += glow * particles * 0.5 * (sin(uniforms.time * 3.0) * 0.2 + 0.8);
    } else if (uniforms.themeIndex < 3.0) {
        // Monochrome theme - white particles
        color.rgb = mix(color.rgb, float3(1.0), particles * 0.9);
    } else {
        // Cosmic theme - nebula effect with particles
        float3 cosmic = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                           noise(uv * 3.0 + uniforms.time * 0.1));
        color.rgb = mix(color.rgb, cosmic, particles * 0.7);
        
        // Add energy pulse effect
        float energyPulse = sin(uniforms.time * 2.0 + noise(uv) * 10.0) * 0.5 + 0.5;
        color.rgb += uniforms.accentColor.rgb * particles * energyPulse * 0.3;
    }
    
    return color;
}

// Neural visualization fragment effects 
float4 neural_fragment_effect(float2 uv, float4 baseColor, constant AudioUniforms &uniforms, float audioIntensity) {
    float4 color = baseColor;
    
    // Use neural parameters to create visualization
    float neuralEnergy = clamp(uniforms.neuralEnergy, 0.1, 1.0);
    float pleasantness = clamp(uniforms.neuralPleasantness, 0.0, 1.0);
    float complexity = clamp(uniforms.neuralComplexity, 0.1, 1.0);
    
    // Calculate center distance for radial effects
    float2 center = float2(0.5, 0.5);
    float2 toCenter = uv - center;
    float distToCenter = length(toCenter);
    float angle = atan2(toCenter.y, toCenter.x);
    
    // Create flowing organic patterns based on neural parameters
    float timeScale = uniforms.time * mix(0.2, 0.5, pleasantness);
    
    // Complexity affects the detail level of the patterns
    float detailLevel = mix(2.0, 8.0, complexity);
    
    // Create base neural pattern
    float pattern = fbm(uv * detailLevel + timeScale * 0.2, int(complexity * 5.0 + 3.0));
    
    // Add beat-responsive elements
    float beatEffect = 0.0;
    if (uniforms.beatDetected > 0.5) {
        // Calculate beat pulse (decaying sine wave)
        float beatTime = fract(uniforms.time * 2.0);
        beatEffect = exp(-beatTime * 5.0) * sin(beatTime * 20.0) * 0.5 + 0.5;
        
        // Radial beat pulse
        float beatRing = smoothstep(0.05, 0.0, abs(distToCenter - beatTime * 0.8));
        pattern += beatRing * 0.3;
    }
    
    // Apply different themes
    if (uniforms.themeIndex < 1.0) {
        // Classic theme - colorful flowing patterns
        float hue = fract(pattern * 2.0 + uniforms.time * 0.1);
        float sat = mix(0.5, 0.9, pleasantness);
        float3 patternColor = hsv2rgb(float3(hue, sat, pattern * neuralEnergy));
        color.rgb = mix(color.rgb, patternColor, 0.7);
        
        // Add beat highlights
        color.rgb += uniforms.accentColor.rgb * beatEffect * 0.3;
    } else if (uniforms.themeIndex < 2.0) {
        // Neon theme - electric neural patterns
        float3 neonBase = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                             sin(angle * 5.0 + uniforms.time) * 0.5 + 0.5);
        
        // Create electric effect with pattern
        float electric = smoothstep(0.4, 0.6, pattern);
        color.rgb = mix(color.rgb, neonBase, electric * neuralEnergy);
        
        // Add glow based on pattern intensity
        color.rgb += neonBase * smoothstep(0.3, 0.7, pattern) * 0.5 * neuralEnergy;
        
        // Add beat pulse glow
        color.rgb += uniforms.accentColor.rgb * beatEffect * 0.5;
    } else if (uniforms.themeIndex < 3.0) {
        // Monochrome theme - stark neural network representation
        float network = step(mix(0.4, 0.7, complexity), pattern);
        
        // Add connecting lines
        float lines = smoothstep(0.05, 0.0, 
            abs(sin(uv.x * 20.0 * complexity + uniforms.time) * 
                sin(uv.y * 20.0 * complexity + uniforms.time * 1.2)));
        
        float combined = max(network, lines * 0.7);
        color.rgb = mix(color.rgb, float3(1.0), combined * neuralEnergy);
        
        // Add beat pulse
        color.rgb *= 1.0 + beatEffect * 0.3;
    } else {
        // Cosmic theme - neural cosmos
        float3 cosmicBase = mix(uniforms.primaryColor.rgb, uniforms.secondaryColor.rgb, 
                               pattern);
        
        // Create nebula-like effect with neural patterns
        float nebula = fbm(uv * mix(3.0, 8.0, complexity) + timeScale * 0.3, 4);
        color.rgb = mix(color.rgb, cosmicBase, nebula * neuralEnergy * 0.8);
        
        // Add energy nodes
        float energyNodes = step(0.7, pattern);
        color.rgb += uniforms.accentColor.rgb * energyNodes * 0.5;
        
        // Add beat expansion waves
        float beatWave = smoothstep(0.1, 0.0, abs(distToCenter - fract(uniforms.time + beatEffect)));
        color.rgb += uniforms.accentColor.rgb * beatWave * beatEffect * 0.5;
    }
    
    return color;
}

// Post-processing effects
float4 apply_post_processing(float4 color, float2 uv, constant AudioUniforms &uniforms) {
    // Apply vignette effect
    float2 center = float2(0.5, 0.5);
    float distFromCenter = length(uv - center);
    float vignette = 1.0 - smoothstep(0.5, 1.5, distFromCenter * 2.0);
    color.rgb *= vignette;
    
    // Apply subtle chromatic aberration for additional visual interest
    if (uniforms.themeIndex > 0.5 && uniforms.themeIndex < 2.5) { // Only for neon and cosmic themes
        float caStrength = 0.01 * uniforms.sensitivity; // Subtle effect
        float2 caOffset = normalize(uv - center) * caStrength;
        
        // Sample with slight offset for different color channels
        float3 caColor = color.rgb;
        caColor.r = color.r * 1.05; // Slight red boost
        caColor.b = mix(color.b, color.b * 0.95, length(caOffset)); // Subtle blue shift
        
        // Mix based on distance from center
        color.rgb = mix(color.rgb, caColor, smoothstep(0.0, 0.8, distFromCenter));
    }
    
    // Apply audio-reactive brightness pulsing
    float pulse = sin(uniforms.time * 2.0) * uniforms.bassLevel * 0.1;
    color.rgb *= 1.0 + pulse;
    
    // Apply final contrast enhancement
    color.rgb = mix(color.rgb, pow(color.rgb, float3(1.1)), 0.3);
    
    // Ensure alpha is set correctly
    color.a = 1.0;
    
    return color;
}

//-----------------------------------------------------------
// Main Fragment Shader
//-----------------------------------------------------------

fragment float4 audio_fragment_shader(VertexOut in [[stage_in]],
                                     constant AudioUniforms &uniforms [[buffer(0)]]) {
    // Extract information from vertex shader
    float2 uv = in.texCoord;
    float4 baseColor = in.color;
    float audioIntensity = in.userData.x;
    int visualizationMode = int(in.userData.y);
    float transitionProgress = in.userData.z;
    float beatDetected = in.userData.w;
    
    // Set default color from vertex shader
    float4 color1 = baseColor;
    float4 color2 = baseColor;
    
    // Apply visualization mode effects
    int prevMode = int(uniforms.previousMode);
    prevMode = clamp(prevMode, 0, 3); // Ensure valid range
    
    // Process current visualization mode
    if (visualizationMode == 0) {
        color1 = spectrum_fragment_effect(uv, baseColor, uniforms, audioIntensity);
    } else if (visualizationMode == 1) {
        color1 = waveform_fragment_effect(uv, baseColor, uniforms, audioIntensity);
    } else if (visualizationMode == 2) {
        color1 = particles_fragment_effect(uv, baseColor, uniforms, audioIntensity);
    } else if (visualizationMode == 3) {
        color1 = neural_fragment_effect(uv, baseColor, uniforms, audioIntensity);
    }
    
    // If in transition, process previous mode too and blend
    if (transitionProgress > 0.0 && transitionProgress < 1.0) {
        if (prevMode == 0) {
            color2 = spectrum_fragment_effect(uv, baseColor, uniforms, audioIntensity);
        } else if (prevMode == 1) {
            color2 = waveform_fragment_effect(uv, baseColor, uniforms, audioIntensity);
        } else if (prevMode == 2) {
            color2 = particles_fragment_effect(uv, baseColor, uniforms, audioIntensity);
        } else if (prevMode == 3) {
            color2 = neural_fragment_effect(uv, baseColor, uniforms, audioIntensity);
        }
        
        // Blend between the two visualization modes
        float4 blendedColor = mix(color2, color1, transitionProgress);
        
        // Add transition effects
        float transitionEffect = sin(transitionProgress * 3.14159) * 0.2;
        blendedColor.rgb += blendedColor.rgb * transitionEffect;
        
        color1 = blendedColor;
    }
    
    // Apply post-processing effects
    float4 finalColor = apply_post_processing(color1, uv, uniforms);
    
    // Mix with background color for proper blending with existing content
    finalColor = mix(uniforms.backgroundColor, finalColor, finalColor.a);
    
    return finalColor;
}
