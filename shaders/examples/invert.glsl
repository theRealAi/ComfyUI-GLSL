/*
@name Invert
@description Invert RGB channels
@version 1.0.0

@input image IMAGE
@input mask MASK

@uniform strength float 1.0 min=0 max=1 step=0.01
*/
vec4 process(vec4 color, ivec2 pixel)
{
    vec3 inverted = 1.0 - color.rgb;
    color.rgb = mix(color.rgb, inverted, params.strength);
    return color;
}
