#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"
#include "common.h"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
    bool rotated;
} params;

layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;

} vOut;

mat4 rotate(float angle) {
    float cos = cos(angle);
    float sin = sin(angle);

    return mat4(
        cos,  0.f,  sin,  0.f,
        0.f,  1.f,  0.f,  0.f,
        -sin, 0.f,  cos,  0.f,
        0.f,  0.f,  0.f,  1.f
    );
}

out gl_PerVertex { vec4 gl_Position; };
void main(void)
{
    const vec4 wNorm = vec4(DecodeNormal(floatBitsToInt(vPosNorm.w)),         0.0f);
    const vec4 wTang = vec4(DecodeNormal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

    mat4 model = params.mModel;
    if (params.rotated)
    {
        model *= rotate(Params.time);
    }

    vOut.wPos     = (model * vec4(vPosNorm.xyz, 1.0f)).xyz;
    vOut.wNorm    = normalize(mat3(transpose(inverse(model))) * wNorm.xyz);
    vOut.wTangent = normalize(mat3(transpose(inverse(model))) * wTang.xyz);
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
