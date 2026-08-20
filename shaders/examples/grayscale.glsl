/*
@name Grayscale
@description Luminance grayscale
@version 1.0.0

@input image IMAGE

@uniform strength float 1.0 min=0 max=1 step=0.01
*/
vec4 process(vec4 color, ivec2 pixel)
{
    float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    color.rgb = mix(color.rgb, vec3(luma), params.strength);
    return color;
}
