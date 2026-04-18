#version 330 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 3) in vec4 a_SkinIndex;   // float indices (0..)
layout(location = 4) in vec4 a_SkinWeight;  // weights sum to 1

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

uniform mat4 u_BindMatrix;
uniform mat4 u_BindMatrixInverse;

// three.js-style bone texture:
// - stores 4 texels (vec4) per bone matrix (mat4 columns)
// - texture is square, size = u_BoneTextureSize
uniform sampler2D u_BoneTexture;
uniform int u_BoneTextureSize;

uniform vec4 u_Color;

out vec4 v_Color;

vec4 boneTexel(int index) {
    int x = index % u_BoneTextureSize;
    int y = index / u_BoneTextureSize;
    return texelFetch(u_BoneTexture, ivec2(x, y), 0);
}

mat4 getBoneMatrix(int boneIndex) {
    int base = boneIndex * 4;
    vec4 c0 = boneTexel(base + 0);
    vec4 c1 = boneTexel(base + 1);
    vec4 c2 = boneTexel(base + 2);
    vec4 c3 = boneTexel(base + 3);
    return mat4(c0, c1, c2, c3);
}

void main() {
    v_Color = u_Color;

    vec4 bindPos = u_BindMatrix * vec4(a_Position, 1.0);
    ivec4 j = ivec4(a_SkinIndex + vec4(0.5)); // round-to-int
    vec4 w = a_SkinWeight;

    vec4 skinned = vec4(0.0);
    skinned += (getBoneMatrix(j.x) * bindPos) * w.x;
    skinned += (getBoneMatrix(j.y) * bindPos) * w.y;
    skinned += (getBoneMatrix(j.z) * bindPos) * w.z;
    skinned += (getBoneMatrix(j.w) * bindPos) * w.w;

    vec4 localPos = u_BindMatrixInverse * skinned;
    vec4 worldPos = u_Model * localPos;

    gl_Position = u_Projection * u_View * worldPos;
}
