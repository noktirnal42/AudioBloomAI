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
    
    // Animation parameters
    float time;                  // Current time in seconds
    float sensitivity;           // Audio sensitivity (0.0-1.0)
    float motionIntensity;       // Motion intensity (0.0-1.0)
    float themeIndex;            // Current theme index (0-3 for Classic, Neon, Monochrome, Cosmic)
};

// Helper functions for visualization effects
float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Get audio sample with smoothing
float getAudioSample(constant AudioUniforms &uniforms, int index, int maxIndex) {
    index = clamp(index, 0, maxIndex);
    return uniforms.audioData[index] * uniforms.sensitivity;
}

// Vertex shader for audio visualization
vertex VertexOut audio_vertex_shader(uint vertexID [[vertex_id]],
                                  constant float4 *positions [[buffer(0)]],
                                  constant AudioUniforms &uniforms [[buffer(1)]]) {
    VertexOut out;
    
    // Get base position from vertex buffer
    float4 basePosition = positions[vertexID];
    float2 texCoord = float2((basePosition.x + 1.0) * 0.5, (basePosition.y + 1.0) * 0.5);
    
    // Calculate vertex distortion based on audio
    float distortionAmount = 0.0;
    
    // Use different audio frequencies based on position
    float xOffset = (texCoord.x * 64);
    float yOffset = (texCoord.y * 64);
    int audioIndex = int(xOffset) % 256;
    
    // Create different distortion effects based on position
    float audioValue = getAudioSample(uniforms, audioIndex, 1023);
    float bassEffect = uniforms.bassLevel * uniforms.motionIntensity * 0.2;
    float timeScale = uniforms.time * 0.5;
    
    // Apply wave-like distortion
    float waveX = sin(texCoord.y * 10.0 + timeScale) * 0.02 * audioValue;
    float waveY = cos(texCoord.x * 10.0 + timeScale) * 0.02 * audioValue;
    
    // Apply pulsing effect from bass
    float pulse = sin(timeScale) * bassEffect;
    
    // Combine effects based on theme
    float themeModifier = 1.0;
    if (uniforms.themeIndex < 0.5) {
        // Classic theme - smooth waves
        distortionAmount = waveX + waveY + pulse * 0.5;
    } else if (uniforms.themeIndex < 1.5) {
        // Neon theme - sharp peaks
        distortionAmount = waveX * 1.5 + pulse * sin(texCoord.x * 30.0);
        themeModifier = 1.5;
    } else if (uniforms.themeIndex < 2.5) {
        // Monochrome theme - subtle motion
        distortionAmount = (waveX + waveY) * 0.7 + pulse * 0.3;
        themeModifier = 0.7;
    } else {
        // Cosmic theme - complex patterns
        distortionAmount = waveX + waveY * sin(texCoord.x * 5.0 + timeScale) + pulse;
        themeModifier = 1.2;
    }
    
    // Apply distortion to position
    out.position = basePosition;
    out.position.x += distortionAmount * themeModifier;
    out.position.y += distortionAmount * themeModifier;
    
    // Calculate color based on position and audio
    float brightness = (audioValue + uniforms.midLevel) * 0.5 + 0.3;
    float hue = fract(texCoord.x + texCoord.y + uniforms.time * 0.05);
    float saturation = uniforms.themeIndex < 2.5 ? 0.7 : 0.5; // Less saturation for Monochrome
    
    // Mix between primary and secondary colors based on audio
    float mixAmount = (sin(texCoord.x * 10.0 + uniforms.time) + 1.0) * 0.5;
    mixAmount = mixAmount * audioValue + 0.2;
    out.color = mix(uniforms.primaryColor, uniforms.secondaryColor, mixAmount);
    
    // Adjust brightness based on audio intensity
    out.color.rgb *= brightness;
    out.texCoord = texCoord;
    
    return out;
}

// Fragment shader for audio visualization
fragment float4 audio_fragment_shader(VertexOut in [[stage_in]],
                                  constant AudioUniforms &uniforms [[buffer(0)]]) {
    float2 uv = in.texCoord;
    float2 center = float2(0.5, 0.5);
    float2 centerDist = uv - center;
    
    // Audio reactivity
    float audioSum = uniforms.bassLevel + uniforms.midLevel + uniforms.trebleLevel;
    
    // Time variables
    float time = uniforms.time;
    float slowTime = time * 0.2;
    
    // Basic color from vertex shader
    float4 baseColor = in.color;
    
    // Different visualization patterns based on theme
    if (uniforms.themeIndex < 0.5) {
        // Classic theme - concentric circles
        float dist = length(centerDist);
        float circle = sin(dist * 20.0 - time * 2.0 + audioSum * 5.0);
        float circleMask = smoothstep(0.0, 0.1, abs(circle));
        baseColor = mix(baseColor, uniforms.primaryColor, circleMask * 0.5);
        
        // Add some rays
        float angle = atan2(centerDist.y, centerDist.x);
        float ray = sin(angle * 10.0 + time);
        baseColor.rgb += uniforms.secondaryColor.rgb * ray * 0.2 * uniforms.trebleLevel;
    }
    else if (uniforms.themeIndex < 1.5) {
        // Neon theme - grid lines
        float gridX = smoothstep(0.03, 0.0, abs(sin(uv.x * 20.0 + time + uniforms.bassLevel * 5.0)));
        float gridY = smoothstep(0.03, 0.0, abs(sin(uv.y * 20.0 + time * 0.7 + uniforms.midLevel * 5.0)));
        float grid = max(gridX, gridY);
        
        // Glow effect
        baseColor.rgb += grid * uniforms.secondaryColor.rgb * 0.7;
        
        // Enhance with bass pulse
        float pulse = sin(time * 2.0) * uniforms.bassLevel;
        baseColor.rgb *= 1.0 + pulse * 0.3;
    }
    else if (uniforms.themeIndex < 2.5) {
        // Monochrome theme - simple bars
        float bars = smoothstep(0.03, 0.0, abs(fract(uv.x * 10.0) - 0.5));
        float barHeight = getAudioSample(uniforms, int(uv.x * 32.0), 31);
        bars *= step(1.0 - barHeight * uniforms.sensitivity * 2.0, uv.y);
        
        // Add static noise texture
        float noise = fract(sin(dot(uv, float2(12.9898, 78.233) * time * 0.01)) * 43758.5453);
        baseColor.rgb = mix(baseColor.rgb, float3(1.0), bars * 0.8);
        baseColor.rgb *= (0.9 + noise * 0.1);
    }
    else {
        // Cosmic theme - nebula effect
        float nebula = sin(uv.x * 5.0 + time) * sin(uv.y * 5.0 + time * 0.7);
        nebula = nebula * 0.5 + 0.5;
        
        // Stars
        float2 starCoord = fract(uv * 20.0);
        float starNoise = fract(sin(dot(floor(uv * 20.0), float2(12.9898, 78.233))) * 43758.5453);
        float star = step(0.98 - uniforms.trebleLevel * 0.1, starNoise);
        float starPulse = sin(time * 5.0 + starNoise * 10.0) * 0.5 + 0.5;
        
        // Combine effects
        baseColor.rgb = mix(baseColor.rgb, uniforms.secondaryColor.rgb, nebula * 0.5);
        baseColor.rgb += star * starPulse * 0.5;
    }
    
    // Add vignette effect
    float vignette = 1.0 - smoothstep(0.5, 1.5, length(centerDist) * 2.0);
    baseColor.rgb *= vignette;
    
    // Mix with background color for proper transparency
    baseColor = mix(uniforms.backgroundColor, baseColor, baseColor.a);
    baseColor.a = 1.0; // Fully opaque
    
    return baseColor;
}</replace>
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Constants for visualization
constant float PI = 3.14159265359;
constant float TWO_PI = 6.28318530718;
constant float HALF_PI = 1.57079632679;

// Music pattern recognition info from neural engine
struct NeuralData {
    float beatIntensity;     // Current beat intensity
    float emotionalValence;  // Emotional positivity (negative to positive)
    float emotionalArousal;  // Emotional intensity/energy
    float complexity;        // Musical complexity measure
    float beatConfidence;    // Confidence level of beat detection
    float patternChange;     // Detects pattern changes in music
    float genre[4];          // Genre classification (electronic, acoustic, etc)
    float reserved[10];      // Reserved for future use
};

// Structure for audio data and visualization parameters
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
    
    // Animation parameters
    float time;                  // Current time in seconds
    float sensitivity;           // Audio sensitivity (0.0-1.0)
    float motionIntensity;       // Motion intensity (0.0-1.0)
    float themeIndex;            // Current theme index (0-3 for Classic, Neon, Monochrome, Cosmic)
    
    // Neural data (available when neural engine is enabled)
    float neuralEnabled;         // Whether neural data is available (0 or 1)
    NeuralData neural;           // Neural engine analysis data
};

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

// Helper functions for visualization effects
float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Get audio sample with smoothing
float getAudioSample(constant AudioUniforms &uniforms, int index, int maxIndex) {
    index = clamp(index, 0, maxIndex);
    return uniforms.audioData[index] * uniforms.sensitivity;
}

// Random and noise functions
float hash(float2 p) {
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    
    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));
    
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Fractal Brownian motion for richer noise
float fbm(float2 p, int octaves, float lacunarity, float gain) {
    float value = 0.0;
    float amplitude = 0.

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
    
    // Animation parameters
    float time;                  // Current time in seconds
    float sensitivity;           // Audio sensitivity (0.0-1.0)
    float motionIntensity;       // Motion intensity (0.0-1.0)
    float themeIndex;            // Current theme index (0-3 for Classic, Neon, Monochrome, Cosmic)
};

// Helper functions for visualization effects
float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Get audio sample with smoothing
float getAudioSample(constant AudioUniforms &uniforms, int index, int maxIndex) {
    index = clamp(index, 0, maxIndex);
    return uniforms.audioData[index] * uniforms.sensitivity;
}

// Vertex shader for audio visualization
vertex VertexOut audio_vertex_shader(uint vertexID [[vertex_id]],
                                    constant float4 *positions [[buffer(0)]],
                                    constant AudioUniforms &uniforms [[buffer(1)]]) {
    VertexOut out;
    
    // Get base position from vertex buffer
    float4 basePosition = positions[vertexID];
    float2 texCoord = float2((basePosition.x + 1.0) * 0.5, (basePosition.y + 1.0) * 0.5);
    
    // Calculate vertex distortion based on audio
    float distortionAmount = 0.0;
    
    // Use different audio frequencies based on position
    float xOffset = (texCoord.x * 64);
    float yOffset = (texCoord.y * 64);
    int audioIndex = int(xOffset) % 256;
    
    // Create different distortion effects based on position
    float audioValue = getAudioSample(uniforms, audioIndex, 1023);
    float bassEffect = uniforms.bassLevel * uniforms.motionIntensity * 0.2;
    float timeScale = uniforms.time * 0.5;
    
    // Apply wave-like distortion
    float waveX = sin(texCoord.y * 10.0 + timeScale) * 0.02 * audioValue;
    float waveY = cos(texCoord.x * 10.0 + timeScale) * 0.02 * audioValue;
    
    // Apply pulsing effect from bass
    float pulse = sin(timeScale) * bassEffect;
    
    // Combine effects based on theme
    float themeModifier = 1.0;
    if (uniforms.themeIndex < 0.5) {
        // Classic theme - smooth waves
        distortionAmount = waveX + waveY + pulse * 0.5;
    } else if (uniforms.themeIndex < 1.5) {
        // Neon theme - sharp peaks
        distortionAmount = waveX * 1.5 + pulse * sin(texCoord.x * 30.0);
        themeModifier = 1.5;
    } else if (uniforms.themeIndex < 2.5) {
        // Monochrome theme - subtle motion
        distortionAmount = (waveX + waveY) * 0.7 + pulse * 0.3;
        themeModifier = 0.7;
    } else {
        // Cosmic theme - complex patterns
        distortionAmount = waveX + waveY * sin(texCoord.x * 5.0 + timeScale) + pulse;
        themeModifier = 1.2;
    }
    
    // Apply distortion to position
    out.position = basePosition;
    out.position.x += distortionAmount * themeModifier;
    out.position.y += distortionAmount * themeModifier;
    
    // Calculate color based on position and audio
    float brightness = (audioValue + uniforms.midLevel) * 0.5 + 0.3;
    float hue = fract(texCoord.x + texCoord.y + uniforms.time * 0.05);
    float saturation = uniforms.themeIndex < 2.5 ? 0.7 : 0.5; // Less saturation for Monochrome
    
    // Mix between primary and secondary colors based on audio
    float mixAmount = (sin(texCoord.x * 10.0 + uniforms.time) + 1.0) * 0.5;
    mixAmount = mixAmount * audioValue + 0.2;
    out.color = mix(uniforms.primaryColor, uniforms.secondaryColor, mixAmount);
    
    // Adjust brightness based on audio intensity
    out.color.rgb *= brightness;
    out.texCoord = texCoord;
    
    return out;
}

// Fragment shader for audio visualization
fragment float4 audio_fragment_shader(VertexOut in [[stage_in]],
                                    constant AudioUniforms &uniforms [[buffer(0)]]) {
    float2 uv = in.texCoord;
    float2 center = float2(0.5, 0.5);
    float2 centerDist = uv - center;
    
    // Audio reactivity
    float audioSum = uniforms.bassLevel + uniforms.midLevel + uniforms.trebleLevel;
    
    // Time variables
    float time = uniforms.time;
    float slowTime = time * 0.2;
    
    // Basic color from vertex shader
    float4 baseColor = in.color;
    
    // Different visualization patterns based on theme
    if (uniforms.themeIndex < 0.5) {
        // Classic theme - concentric circles
        float dist = length(centerDist);
        float circle = sin(dist * 20.0 - time * 2.0 + audioSum * 5.0);
        float circleMask = smoothstep(0.0, 0.1, abs(circle));
        baseColor = mix(baseColor, uniforms.primaryColor, circleMask * 0.5);
        
        // Add some rays
        float angle = atan2(centerDist.y, centerDist.x);
        float ray = sin(angle * 10.0 + time);
        baseColor.rgb += uniforms.secondaryColor.rgb * ray * 0.2 * uniforms.trebleLevel;
    }
    else if (uniforms.themeIndex < 1.5) {
        // Neon theme - grid lines
        float gridX = smoothstep(0.03, 0.0, abs(sin(uv.x * 20.0 + time + uniforms.bassLevel * 5.0)));
        float gridY = smoothstep(0.03, 0.0, abs(sin(uv.y * 20.0 + time * 0.7 + uniforms.midLevel * 5.0)));
        float grid = max(gridX, gridY);
        
        // Glow effect
        baseColor.rgb += grid * uniforms.secondaryColor.rgb * 0.7;
        
        // Enhance with bass pulse
        float pulse = sin(time * 2.0) * uniforms.bassLevel;
        baseColor.rgb *= 1.0 + pulse * 0.3;
    }
    else if (uniforms.themeIndex < 2.5) {
        // Monochrome theme - simple bars
        float bars = smoothstep(0.03, 0.0, abs(fract(uv.x * 10.0) - 0.5));
        float barHeight = getAudioSample(uniforms, int(uv.x * 32.0), 31);
        bars *= step(1.0 - barHeight * uniforms.sensitivity * 2.0, uv.y);
        
        // Add static noise texture
        float noise = fract(sin(dot(uv, float2(12.9898, 78.233) * time * 0.01)) * 43758.5453);
        baseColor.rgb = mix(baseColor.rgb, float3(1.0), bars * 0.8);
        baseColor.rgb *= (0.9 + noise * 0.1);
    }
    else {
        // Cosmic theme - nebula effect
        float nebula = sin(uv.x * 5.0 + time) * sin(uv.y * 5.0 + time * 0.7);
        nebula = nebula * 0.5 + 0.5;
        
        // Stars
        float2 starCoord = fract(uv * 20.0);
        float starNoise = fract(sin(dot(floor(uv * 20.0), float2(12.9898, 78.233))) * 43758.5453);
        float star = step(0.98 - uniforms.trebleLevel * 0.1, starNoise);
        float starPulse = sin(time * 5.0 + starNoise * 10.0) * 0.5 + 0.5;
        
        // Combine effects
        baseColor.rgb = mix(baseColor.rgb, uniforms.secondaryColor.rgb, nebula * 0.5);
        baseColor.rgb += star * starPulse * 0.5;
    }
    
    // Add vignette effect
    float vignette = 1.0 - smoothstep(0.5, 1.5, length(centerDist) * 2.0);
    baseColor.rgb *= vignette;
    
    // Mix with background color for proper transparency
    baseColor = mix(uniforms.backgroundColor, baseColor, baseColor.a);
    baseColor.a = 1.0; // Fully opaque
    
    return baseColor;
}

