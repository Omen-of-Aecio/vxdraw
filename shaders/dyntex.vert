#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec2 v_dxdy;
layout(location = 3) in float rotation;
layout(location = 4) in float scale;
layout(location = 5) in vec4 color;

layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec4 f_color;

layout(push_constant) uniform PushConstant {
    mat4 view;
} push_constant;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    mat2 rotmatrix = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    vec2 pos = rotmatrix * scale * v_pos.xy;
    f_uv = v_uv;
    f_color = color;
    gl_Position = push_constant.view * vec4(pos + v_dxdy, v_pos.z, 1.0);
}
