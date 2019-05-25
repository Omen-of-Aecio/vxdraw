#version 450
#extension GL_ARG_separate_shader_objects : enable

layout (location = 0) in vec2 pos;
layout (location = 0) out vec2 texpos;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    texpos = (pos + 1)/2;
    gl_Position = vec4(pos, 0, 1);
}
