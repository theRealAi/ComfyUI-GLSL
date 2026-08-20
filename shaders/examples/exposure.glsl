/*
@name Exposure
@description Exposure adjustment in stops
@version 1.0.0

@input image IMAGE

@uniform exposure float 0.0 min=-10 max=10 step=0.01
*/
vec4 process(vec4 color, ivec2 pixel)
{
    color.rgb *= exp2(params.exposure);
    return color;
}
