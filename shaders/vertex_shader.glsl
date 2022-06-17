#version 430
layout(location = 0) in vec2 position;
layout(location = 1) in float height;

mat4 buildTranslate(float x, float y, float z) {
    mat4 trans = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0);
    return trans;
}

mat4 buildRotateX(float rad) {
    mat4 xrot = mat4(1.0, 0.0, 0.0, 0.0, 0.0, cos(rad), -sin(rad), 0.0, 0.0, sin(rad), cos(rad), 0.0, 0.0, 0.0, 0.0, 1.0);
    return xrot;
}

mat4 buildRotateY(float rad) {
    mat4 yrot = mat4(cos(rad), 0.0, sin(rad), 0.0, 0.0, 1.0, 0.0, 0.0, -sin(rad), 0.0, cos(rad), 0.0, 0.0, 0.0, 0.0, 1.0);
    return yrot;
}
// builds and returns a matrix that performs a rotation around the Z axis
mat4 buildRotateZ(float rad) {
    mat4 zrot = mat4(cos(rad), -sin(rad), 0.0, 0.0, sin(rad), cos(rad), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    return zrot;
}

// mat4 buildScale(float x, float y, float z) {
//     mat4 scale = mat4(x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 1.0);
//     return scale;
// }

uniform mat4 m_matrix;
uniform mat4 v_matrix;
uniform mat4 proj_matrix;
uniform float tf;

out vec4 varyingColor;

void main(void) {
    gl_Position = proj_matrix * v_matrix * vec4(position, height, 1.0);
}