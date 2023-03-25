#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#include "common.h"

layout (triangles) in;
layout (triangle_strip, max_vertices = 6) out;

layout (push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout (location = 0) in VS_IN
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;
} vIn[];

layout (location = 0) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;
} vOut;

layout (binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

void emitVertex(vec3 pos, vec3 norm, vec3 tangent, vec2 texCoord) {
    gl_Position = params.mProjView * vec4(pos, 1.0f);
    vOut.wPos     = pos;
    vOut.wNorm    = norm;
    vOut.wTangent = tangent;
    vOut.texCoord = texCoord;
    EmitVertex();
}


void main(void)
{
    float magnitude = 0.15;
    vec3 triangleNorm = normalize(vIn[0].wNorm + vIn[1].wNorm + vIn[2].wNorm);
    vec3 avgVertexPos = (vIn[0].wPos + vIn[1].wPos + vIn[2].wPos) / 3.0f;
    vec3 avgVertexTangent = normalize(vIn[0].wTangent + vIn[1].wTangent + vIn[2].wTangent);
    vec2 avgVertexTexCoord = (vIn[0].texCoord + vIn[1].texCoord + vIn[2].texCoord) / 3.0f;

    avgVertexPos += magnitude * (1 + cos(2.0 * Params.time + avgVertexPos.x)) * triangleNorm ;

    emitVertex(vIn[0].wPos, vIn[0].wNorm, vIn[0].wTangent, vIn[0].texCoord);
    emitVertex(vIn[1].wPos, vIn[1].wNorm, vIn[1].wTangent, vIn[1].texCoord);
    emitVertex(avgVertexPos, triangleNorm, avgVertexTangent, avgVertexTexCoord);
    emitVertex(vIn[2].wPos, vIn[2].wNorm, vIn[2].wTangent, vIn[2].texCoord);
    emitVertex(vIn[0].wPos, vIn[0].wNorm, vIn[0].wTangent, vIn[0].texCoord);
    emitVertex(avgVertexPos, triangleNorm, avgVertexTangent, avgVertexTexCoord);
    emitVertex(vIn[2].wPos, vIn[2].wNorm, vIn[2].wTangent, vIn[2].texCoord);
    emitVertex(vIn[1].wPos, vIn[1].wNorm, vIn[1].wTangent, vIn[1].texCoord);

    EndPrimitive();
}