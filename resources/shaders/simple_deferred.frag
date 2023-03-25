#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 norm;
layout(location = 1) out vec4 albedoSpec;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
    uint id_albedo;
} PushConstant;

layout (location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

void main()
{
  norm = vec4(0.5f*normalize(surf.wNorm) + vec3(0.5f), 1.0);
  switch(PushConstant.id_albedo)
  {
  case 0:
    albedoSpec = vec4(0.f, 1.f, 1.f, 1.f);
    break;
  case 1:
    albedoSpec = vec4(1.f, 0.f, 1.f, 1.f);
    break;
  case 2:
    albedoSpec = vec4(0.f, 0.f, 1.f, 1.f);
    break;
  case 3:
    albedoSpec = vec4(1.f, 1.f, 0.f, 1.f);
    break;
  case 4:
    albedoSpec = vec4(0.f, 1.f, 0.f, 1.f);
    break;
  case 5:
    albedoSpec = vec4(1.f, 0.f, 0.f, 1.f);
    break;
  }
}