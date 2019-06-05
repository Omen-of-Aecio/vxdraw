#version 450

layout (location = 0) in vec2 texpos;
layout (location = 0) out vec4 Color;

layout(push_constant) uniform PushConstant {
    layout (offset = 0) vec4 rand_seed;
} push;

// Hash function: http://amindforeverprogramming.blogspot.com/2013/07/random-floats-in-glsl-330.html
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
uint hash(uvec3 v) {
    return hash( v.x ^ hash(v.y) ^ hash(v.z) );
}
float random(uvec3 pos) {
    const uint mantissaMask = 0x007FFFFFu;
    const uint one          = 0x3F800000u;
   
    uint h = hash( pos );
    h &= mantissaMask;
    h |= one;
    
    float  r2 = uintBitsToFloat( h );
    return r2 - 1.0;
}
float random(vec3 pos) {
    return random(floatBitsToUint(pos));
}
// returns fraction part
float separate(float n, out float i) {
    float frac = modf(n, i);
    if (n < 0.f) {
        frac = 1 + frac; // make fraction non-negative and invert (1 - frac)
        i --;
    }
    return frac;
}

// Perlin: http://www.iquilezles.org/www/articles/morenoise/morenoise.htm
float perlin(vec3 pos, out float dnx, out float dny, out float dnz) {
    float i, j, k;
    float u, v, w;

    // Separate integer and fractional part of coordinates
    u = separate( pos.x, i);
    v = separate( pos.y, j);
    w = separate( pos.z, k);


    float du = 30.0f*u*u*(u*(u-2.0f)+1.0f);
    float dv = 30.0f*v*v*(v*(v-2.0f)+1.0f);
    float dw = 30.0f*w*w*(w*(w-2.0f)+1.0f);

    u = u*u*u*(u*(u*6.0f-15.0f)+10.0f);
    v = v*v*v*(v*(v*6.0f-15.0f)+10.0f);
    w = w*w*w*(w*(w*6.0f-15.0f)+10.0f);

    float a = random( vec3(i+0, j+0, k+0) );
    float b = random( vec3(i+1, j+0, k+0) );
    float c = random( vec3(i+0, j+1, k+0) );
    float d = random( vec3(i+1, j+1, k+0) );
    float e = random( vec3(i+0, j+0, k+1) );
    float f = random( vec3(i+1, j+0, k+1) );
    float g = random( vec3(i+0, j+1, k+1) );
    float h = random( vec3(i+1, j+1, k+1) );

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    /* dnx = du * (k1 + k4*v + k6*w + k7*v*w); */
    /* dny = dv * (k2 + k5*w + k4*u + k7*w*u); */
    /* dnz = dw * (k3 + k6*u + k5*v + k7*u*v); */
    return k0 + k1*u + k2*v + k3*w + k4*u*v + k5*v*w + k6*w*u + k7*u*v*w;
}

// Note: It starts (octave 1) with the highest frequency, `width`
float FBM(vec3 pos, int octaves) {
    float a, b, c;
    float result = 0;
    float p;

    pos *= push.rand_seed.x; // Frequency = pixel
    /* pos *= 1000; */

    const float power = 3;  // Higher -> lower frequencies dominate. Normally 2.
    float pos_factor = 1.f;
    float strength_factor = 1.f / pow(power, octaves);
    for (int i = 0; i < octaves; i ++)
    {
        p = perlin(pos * pos_factor, a, b, c );
        result += (power - 1) * strength_factor * p;

        pos_factor *= 0.5f;
        strength_factor *= power;
    }

    return result;
}

void main()
{
    int octaves = 8;
    float r;
    r = FBM(vec3(texpos,0) + push.rand_seed.yzw, octaves);
    r = step(0.5, r);
    Color = vec4(vec3(r), 1);
}
