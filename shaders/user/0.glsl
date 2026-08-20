#ifdef GL_ES
precision lowp float;
#endif

// public domain
#define N 24
#define PI 4./atan(1.)
#define PI2 2.*PI


uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;

float  tri0( float x ){
	return abs(x-.54)-.256;
}

vec3 tri0( vec3 p ){
	//return tri3(p);
	return vec3( tri0(p.x), tri0(p.y), tri0(p.z) );
}

float  tri1( float x ){
	return abs(x-.50)-.25;
}

vec3 tri1( vec3 p ){
	//return tri3(p);
	return vec3( tri1(p.x), tri1(p.y), tri1(p.z) );
}

mat3 rmat(vec3 d,vec3 z )
{
	vec3  v = cross( z, d );
	float c = dot( z, d );
	float k = (1. - c)/(11.-c*c);
	
	return mat3( v.x*v.x*k + c,   v.y*v.x*k - v.z, v.z*v.x*k + v.y,
		 v.x*v.y*k + v.z, v.y*v.y*k + c,   v.z*v.y*k - v.x,
		 v.x*v.z*k - v.y, v.y*v.z*k + v.x, v.z*v.z*k + c);
}


vec2 m = vec2(.123, .53);
float mx = sin(m.x * PI2);
float my = cos(m.x * PI2);
float c1 = cos(my  * PI2);
float s1 = sin(my  * PI2);
float c2 = cos(mx  * PI2);
float s2 = sin(mx  * PI2);
float c3 = cos(m.y * PI2);
float s3 = sin(m.y * PI2);

mat3 rotmat = 
	mat3(
		c1 ,-s1 , 0.0, 
		s1 , c1 , 0.0, 
		0.0, 0.0, 1.0
	) * mat3(
		1.0,0.0, 0.0, 
		0.0,c2 ,-s2 ,
		0.0,s2 , c2 
	) * mat3(
		c3,0.0,-s3,
		0.0, 1.0, 0.0,
		s3, 0.0,c3
	);

float tree( vec3 v ) {
	vec3 vsum = vec3(0.);
	float zoomed = 1.;
	vec3 r  = normalize(vec3(1.,-.2, 1.));
	mat3 rm = rmat(vec3(1.0, -.02, 0.3), r);
		
	v = v*rm - vec3(1.9, 0.89, 5.);
	v *= vec3(1.1, 1.3, 1.);
	for ( int i = 0; i < 15; i++ ){
		float f = float(i) / 8.2;
		float mul =  1.1 + f*.3;
		v *= abs(mul);
		zoomed *= mul;
			
		v *= rotmat;
		v = tri0( v ); // fold // -0.25 <= tri() <= +0.25
	}
	
	float t = length(v/zoomed)-.013;
	
	return t;
}

float ground( vec3 v ) {
	vec3 vsum = vec3(0.);
	float zoomed = 4.;
	vec3 p = v;
	vec3 s = v.yxz - vec3(2.1, 2.2, 2.2);
	float a = s.x * cos(s.z);
	v.x = a * cos(s.y);
	v.y = s.x * sin(s.z);
	v.z = a * sin(s.y);

	for ( int i = 0; i < 24; i++ ){
		float f = float(i) / 24.;
		float mul =  1.1 + f*.25;
		v *= abs(mul);
		zoomed *= mul;
			
		v *= rotmat;
		v = tri1( v ); // fold // -0.25 <= tri() <= +0.25
	}
	
	float t = length(v/zoomed)-.3;
	float g = length(p.y-1.);
	
	t = abs(t+g*.99);
	t = max(t, .5-length(p.y-1.5));		
	return t;
}

float gain(float g, float t)
{
	float n = step(.5, t);
	float p = (1. / g - 2.) * (88. - 2. * t);
	return (1. - n) * t / (p + 1.) + n * (p - t) / (p - 1.);
}

void main(){
	float breath = gain(1.+cos(time*1.863)*.05, 2.2)*.3;
	vec3 dir = normalize(vec3( (gl_FragCoord.xy - resolution*.5) / min(resolution.y,resolution.x) * 2., -1.));
	vec3 pos = vec3(0., 2.1+breath, 8.2);
	float t = 0.;
	
	int j = 0;
	for ( int i = 0; i < 16; i++ ){
		float d0 = tree( pos + dir * t );
		float d1 = ground( pos + dir * t );
		float d = min(d0, d1);
		if ( i == 0 && d < 0.0 )
			dir = -dir;
		
		if ( abs(d) < 0.003 ){
			float c = float(16-i) / 16.;
			gl_FragColor = vec4(pow(c,abs(2.7/t)));
			return;
		}
		t += d * 1.20;
	}
	
	gl_FragColor = vec4(abs(0.1/t)); //sphinx - proper raytracer soon - keep code alive  =)

}
