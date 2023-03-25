#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>
#include <random>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment
  });

  shadowMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });

  m_uboMappedMem = constants.map();

  matricesTransform = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size        = sizeof(LiteMath::float4x4) * m_instNumber,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "matricesTransform"
  });

  m_uboMatricesTransform = matricesTransform.map();

  visibleObjNum = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size        = sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "visibleObjNum"
  });

  m_uboVisibleObjNum = visibleObjNum.map();

  visibleObjIds = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size        = sizeof(uint32_t) * m_instNumber,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "visibleObjIds"
  });

  m_uboVisibleObjIds = visibleObjIds.map();

  buffIndirect = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size        = sizeof(VkDrawIndexedIndirectCommand) * 2,
    .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "buffIndirect"
  });

  m_uboBuffIndirect = buffIndirect.map();
}

void SimpleShadowmapRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  m_pScnMgr->LoadSceneXML(path, transpose_inst_matrices);

  // TODO: Make a separate stage
  loadShaders();
  PreparePipelines();

  auto loadedCam = m_pScnMgr->GetCamera(0);
  m_cam.fov = loadedCam.fov;
  m_cam.pos = float3(loadedCam.pos);
  m_cam.up  = float3(loadedCam.up);
  m_cam.lookAt = float3(loadedCam.lookAt);
  m_cam.tdist  = loadedCam.farPlane;
}

void SimpleShadowmapRender::DeallocateResources()
{
  mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  shadowMap.reset();
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);  

  constants = etna::Buffer();
}





/// PIPELINES CREATION

void SimpleShadowmapRender::PreparePipelines()
{
  // create full screen quad for debug purposes
  // 
  m_pFSQuad = std::make_shared<vk_utils::QuadRenderer>(0,0, 512, 512);
  m_pFSQuad->Create(m_context->getDevice(),
    VK_GRAPHICS_BASIC_ROOT "/resources/shaders/quad3_vert.vert.spv",
    VK_GRAPHICS_BASIC_ROOT "/resources/shaders/quad.frag.spv",
    vk_utils::RenderTargetInfo2D{
      .size          = VkExtent2D{ m_width, m_height },// this is debug full screen quad
      .format        = m_swapchain.GetFormat(),
      .loadOp        = VK_ATTACHMENT_LOAD_OP_LOAD,// seems we need LOAD_OP_LOAD if we want to draw quad to part of screen
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
    }
  );
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("frustum_culling", { VK_GRAPHICS_BASIC_ROOT "/resources/shaders/frustum_culling.comp.spv" });
  etna::create_program("simple_material",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_shadow.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
}

void SimpleShadowmapRender::SetupSimplePipeline()
{
  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     2},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             3}
  };

  m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(m_context->getDevice(), dtypes, 2);
  
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, matricesTransform.get());
  m_pBindings->BindBuffer(1, visibleObjNum.get());
  m_pBindings->BindBuffer(2, visibleObjIds.get());
  m_pBindings->BindEnd(&m_fustrumCullingDS, &m_fustrumCullingDSLayout);

  m_pBindings->BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pBindings->BindImage(0, shadowMap.getView({}), defaultSampler.get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindEnd(&m_quadDS, &m_quadDSLayout);

  etna::VertexShaderInputDescription sceneVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription()
        }}
    };

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_basicForwardPipeline = pipelineManager.createGraphicsPipeline("simple_material",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
  m_shadowPipeline = pipelineManager.createGraphicsPipeline("simple_shadow",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });
  m_frustumCullingPipeline = pipelineManager.createComputePipeline("frustum_culling", {});
}

void SimpleShadowmapRender::DestroyPipelines()
{
  m_pFSQuad     = nullptr; // smartptr delete it's resources
}

void SimpleShadowmapRender::RunFrustumCullingShader(VkCommandBuffer a_cmdBuff, const float4x4 &a_wvp)
{
  std::mt19937 generator;
  generator.seed(63);
  std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
  std::vector<LiteMath::float4x4> positions;
  positions.reserve(m_instNumber);

  for (size_t i = 1; i <= m_instNumber; i++)
  {
    auto x = dist(generator);
    auto y = 0.0f;
    auto z = dist(generator);

    auto newPosition = LiteMath::translate4x4({ x, y, z });
    positions.push_back(newPosition);
  }

  memcpy(m_uboMatricesTransform, positions.data(), sizeof(LiteMath::float4x4) * positions.size());
  vkCmdFillBuffer(a_cmdBuff, visibleObjNum.get(), 0, VK_WHOLE_SIZE, 0);
  vkCmdFillBuffer(a_cmdBuff, visibleObjIds.get(), 0, VK_WHOLE_SIZE, 0);
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_COMPUTE_BIT);
  pushConstComp.projView        = a_wvp;

  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    pushConstComp.instNumber = m_instNumber;
    pushConstComp.box               = m_pScnMgr->GetInstanceBbox(i);
    vkCmdPushConstants(
      a_cmdBuff, m_frustumCullingPipeline.getVkPipelineLayout(), stageFlags, 0, sizeof(pushConstComp), &pushConstComp);
    vkCmdDispatch(a_cmdBuff, pushConstComp.instNumber / 32 + 1, 1, 1);
    {
      std::array barriers
      {
        VkBufferMemoryBarrier2
        {
          .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
          .srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
          .dstStageMask  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
          .buffer        = visibleObjNum.get(),
          .size          = sizeof(int32_t)
        }
      };
      VkDependencyInfo depInfo
      {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .dependencyFlags          = VK_DEPENDENCY_BY_REGION_BIT,
        .bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pBufferMemoryBarriers    = barriers.data(),
      };
      vkCmdPipelineBarrier2(a_cmdBuff, &depInfo);
    }
  }
}

/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuf  = m_pScnMgr->GetIndexBuffer();
  
  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  pushConst2M.projView = a_wvp;
  pushConst2M.model    = m_pScnMgr->GetInstanceMatrix(0);

  VkDrawIndexedIndirectCommand *buffIndirCmd = (VkDrawIndexedIndirectCommand *)m_uboBuffIndirect;

  MeshInfo mesh_info               = m_pScnMgr->GetMeshInfo(m_pScnMgr->GetInstanceInfo(0).mesh_id);
  buffIndirCmd->firstIndex         = mesh_info.m_indexOffset;
  buffIndirCmd->firstInstance      = 0;
  buffIndirCmd->indexCount         = mesh_info.m_indNum;
  buffIndirCmd->instanceCount      = *(uint *)m_uboVisibleObjNum;
  buffIndirCmd->vertexOffset       = mesh_info.m_vertexOffset;
  vkCmdPushConstants(a_cmdBuff, m_basicForwardPipeline.getVkPipelineLayout(), stageFlags, 0, sizeof(pushConst2M), &pushConst2M);
  vkCmdDrawIndexedIndirect(a_cmdBuff, buffIndirect.get(), 0, m_pScnMgr->InstancesNum(), sizeof(VkDrawIndexedIndirectCommand));
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  {
    auto frustumCullingInfo = etna::get_shader_program("frustum_culling");
    auto set                = etna::create_descriptor_set(frustumCullingInfo.getDescriptorLayoutId(0), a_cmdBuff,
        { 
          etna::Binding{ 0, matricesTransform.genBinding() },
          etna::Binding{ 1, visibleObjNum.genBinding() },
          etna::Binding{ 2, visibleObjIds.genBinding() }
        });
    VkDescriptorSet vkSet = set.getVkSet();
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_frustumCullingPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_frustumCullingPipeline.getVkPipeline());
    RunFrustumCullingShader(a_cmdBuff, m_worldViewProj);
  }

  //// draw scene to shadowmap
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, { 2048, 2048 }, {}, { VK_NULL_HANDLE, shadowMap.getView({}) });
    {
      auto simpleShadowInfo = etna::get_shader_program("simple_shadow");
      auto set              = etna::create_descriptor_set(simpleShadowInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding{ 0, constants.genBinding() },
        etna::Binding{ 2, matricesTransform.genBinding() },
        etna::Binding{ 3, visibleObjIds.genBinding() },
      });
      VkDescriptorSet vkSet = set.getVkSet();
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
      DrawSceneCmd(a_cmdBuff, m_lightMatrix);
    }
  }

  //// draw final scene to screen
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding{ 0, constants.genBinding() },
      etna::Binding{ 1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal) },
      etna::Binding{ 2, matricesTransform.genBinding() },
      etna::Binding{ 3, visibleObjIds.genBinding() }
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {{a_targetImage, a_targetImageView}}, mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicForwardPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_basicForwardPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
  }

  if(m_input.drawFSQuad)
  {
    float scaleAndOffset[4] = {0.5f, 0.5f, -0.5f, +0.5f};
    m_pFSQuad->SetRenderTarget(a_targetImageView);
    m_pFSQuad->DrawCmd(a_cmdBuff, m_quadDS, scaleAndOffset);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
