#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec4 f_color;

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform texture2D f_texture;
layout(set = 0, binding = 1) uniform sampler f_sampler;

void main() {
    color = texture(sampler2D(f_texture, f_sampler), f_uv);
    color.a *= f_color.a;
    color.rgb += f_color.rgb;
}
