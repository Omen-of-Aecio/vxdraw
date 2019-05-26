#version 450
#extension GL_ARG_separate_shader_objects : enable
layout (location = 0) in vec2 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 dxdy;
layout (location = 3) in float rotation;
layout (location = 4) in float scale;

layout(push_constant) uniform PushConstant {
    float w_over_h;
} push_constant;

layout (location = 0) out vec4 outcolor;
out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    mat2 rotmatrix = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    vec2 pos = rotmatrix * scale * position;
    if (push_constant.w_over_h >= 1.0) {
        pos.x /= push_constant.w_over_h;
    } else {
        pos.y *= push_constant.w_over_h;
    }
    gl_Position = vec4(pos + dxdy, 0.0, 1.0);
    outcolor = color;
}
