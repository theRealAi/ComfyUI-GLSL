// SPDX-FileCopyrightText: 2021 Martin J. Fiedler <keyj@emphy.de>
// SPDX-License-Identifier: MIT

// @gips_version=1

uniform float ev;           // @min=-5 @max=5 EV
uniform float gamma = 2.2;  // @min=.5 @max=10 working gamma
uniform float gamutMap;     // preserve hue in clipped regions
uniform float reinhard;     // @switch brighten with Reinhard tone compression
uniform float clipMark;     // @switch mark clipped regions

vec3 run(vec3 rgb) {
    // forward gamma
    rgb = pow(rgb, vec3(gamma));

    // apply gain
    float gain = exp2(ev);
    rgb *= gain;

    // tone mapping
    if (reinhard > 0.5) {
        rgb = rgb / (rgb + 1.0);
        // post-scale so white stays white
        rgb *= (gain + 1.0) / gain;
    }

    // gamut mapping
    float minRGB = min(min(rgb.r, rgb.g), rgb.b);
    float maxRGB = max(max(rgb.r, rgb.g), rgb.b);
    if ((maxRGB > 1.0) && (minRGB < maxRGB) && (gamutMap > 0.0)) {
        rgb = mix(rgb, vec3(minRGB) + (rgb - vec3(minRGB)) * vec3((1.0 - minRGB) / (maxRGB - minRGB)), gamutMap);
    }

    // reverse gamma
    rgb = pow(rgb, vec3(1.0 / gamma));

    // mark clipped pixels
    if (clipMark > 0.5) {
        if (maxRGB >= (254.0/255.0)) { return vec3(1.0, 0.0, 0.0); }
        if (minRGB <=   (1.0/255.0)) { return vec3(0.0, 0.0, 1.0); }
    }

    return rgb;
}
