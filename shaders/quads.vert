#version 450
#extension GL_ARG_separate_shader_objects : enable
layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 dxdy;
layout (location = 3) in float rotation;
layout (location = 4) in float scale;

layout(push_constant) uniform PushConstant {
    mat4 view;
} push_constant;

layout (location = 0) out vec4 outcolor;

out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
    mat2 rotmatrix = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    vec2 pos = rotmatrix * scale * position.xy;
    gl_Position = push_constant.view * vec4(pos + dxdy, position.z, 1.0);
    outcolor = color;
}
