// SPDX-FileCopyrightText: 2024 Martin J. Fiedler <keyj@emphy.de>
// SPDX-License-Identifier: MIT

// @gips_version=1

uniform float ldC   = 0.0;  // @min=-2 @max=2 contrast adjustment
uniform float pivot = 0.5;  // mean brightness
uniform float gmap  = 1.0;  // gamut mapping
uniform float gamma = 2.2;  // @min=.2 @max=5 working gamma

vec3 run(vec3 rgb) {
    float exponent = -log2(pivot);
    float contrast = exp2(ldC);

    rgb = pow(rgb, vec3(gamma));
    float origLuma = dot(rgb, vec3(0.25, 0.5, 0.25));

    float newLuma = pow(origLuma, 1.0 / exponent);
    bool upper = (newLuma > 0.5);
    if (upper) { newLuma = 1.0 - newLuma; }
    newLuma = 0.5 * pow(2.0 * newLuma, contrast);
    if (upper) { newLuma = 1.0 - newLuma; }
    newLuma = pow(newLuma, exponent);

    rgb *= newLuma / origLuma;
    float minRGB = min(min(rgb.r, rgb.g), rgb.b);
    float maxRGB = max(max(rgb.r, rgb.g), rgb.b);
    if ((maxRGB > 1.0) && (minRGB < maxRGB)) {
        rgb = mix(rgb, vec3(minRGB) + (rgb - vec3(minRGB)) * vec3((1.0 - minRGB) / (maxRGB - minRGB)), gmap);
    }

    return pow(rgb, vec3(1.0 / gamma));
}
