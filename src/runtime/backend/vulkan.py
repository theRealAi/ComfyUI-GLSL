"""
Vulkan compute backend for GLSL image processing.

Uses host-visible staging buffers for portable PyTorch IMAGE upload/download.
Works on discrete and integrated GPUs (NVIDIA/AMD/Intel).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComputePipeline:
    pipeline: Any
    pipeline_layout: Any
    descriptor_set_layout: Any
    shader_module: Any
    push_constant_size: int


@dataclass
class StorageImage:
    image: Any
    memory: Any
    view: Any
    width: int
    height: int


class VulkanBackend:
    """Vulkan compute backend with pipeline cache and staging I/O."""

    def __init__(self):
        self.instance = None
        self.device = None
        self.physical_device = None
        self.queue = None
        self.command_pool = None
        self.descriptor_pool = None
        self.compute_queue_family_index = 0
        self.gpu_info: Dict[str, Any] = {}
        self._vulkan_available = False
        self._vk = None
        self._pipeline_cache: Dict[str, ComputePipeline] = {}
        self._initialized = False

    def is_available(self) -> bool:
        return self._vulkan_available and self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            import vulkan as vk
        except ImportError as e:
            raise Exception("Vulkan Python bindings not installed (pip install vulkan)") from e

        self._vk = vk
        self._vulkan_available = True

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="ComfyUI GLSL",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="ComfyUI-GLSL",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not physical_devices:
            raise Exception("No Vulkan-compatible GPU found")

        self.physical_device = self._select_device(physical_devices)
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        device_name = props.deviceName
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        self.gpu_info = {
            "device_name": device_name,
            "api_version": props.apiVersion,
            "driver_version": props.driverVersion,
            "vendor_id": props.vendorID,
            "device_id": props.deviceID,
            "device_type": int(props.deviceType),
        }

        family = self._find_compute_queue_family(self.physical_device)
        if family < 0:
            raise Exception("No compute queue family found")
        self.compute_queue_family_index = family

        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, family, 0)

        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

        desc_pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=64,
            poolSizeCount=1,
            pPoolSizes=[
                vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=128,
                )
            ],
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, desc_pool_info, None)
        self._initialized = True
        logger.info("Vulkan initialized on %s", device_name)

    def _select_device(self, devices):
        vk = self._vk
        discrete = []
        integrated = []
        other = []
        for d in devices:
            t = vk.vkGetPhysicalDeviceProperties(d).deviceType
            if t == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                discrete.append(d)
            elif t == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                integrated.append(d)
            else:
                other.append(d)
        for group in (discrete, integrated, other):
            if group:
                return group[0]
        return devices[0]

    def _find_compute_queue_family(self, physical_device) -> int:
        vk = self._vk
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        for i, fam in enumerate(families):
            if fam.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                return i
        return -1

    def get_gpu_info(self) -> Dict[str, Any]:
        return dict(self.gpu_info)

    def get_or_create_pipeline(self, cache_key: str, spirv: bytes, push_constant_size: int) -> ComputePipeline:
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        pipeline = self._create_compute_pipeline(spirv, push_constant_size)
        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def _create_compute_pipeline(self, spirv: bytes, push_constant_size: int) -> ComputePipeline:
        vk = self._vk
        if len(spirv) % 4 != 0:
            raise Exception("SPIR-V size must be a multiple of 4")

        # vulkan package casts pCode via ffi.from_buffer; pass raw bytes
        module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spirv),
            pCode=spirv,
        )
        shader_module = vk.vkCreateShaderModule(self.device, module_info, None)

        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
        ]
        set_layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, set_layout_info, None)

        push_range = None
        push_count = 0
        if push_constant_size > 0:
            push_range = [
                vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=push_constant_size,
                )
            ]
            push_count = 1

        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=push_count,
            pPushConstantRanges=push_range,
        )
        pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)
        pipeline = pipelines[0]

        return ComputePipeline(
            pipeline=pipeline,
            pipeline_layout=pipeline_layout,
            descriptor_set_layout=descriptor_set_layout,
            shader_module=shader_module,
            push_constant_size=push_constant_size,
        )

    def _find_memory_type(self, type_bits: int, properties: int) -> int:
        vk = self._vk
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise Exception("Failed to find suitable Vulkan memory type")

    def _create_buffer(self, size: int, usage: int, properties: int) -> Tuple[Any, Any]:
        vk = self._vk
        info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = vk.vkCreateBuffer(self.device, info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=self._find_memory_type(reqs.memoryTypeBits, properties),
        )
        memory = vk.vkAllocateMemory(self.device, alloc, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    def _create_storage_image(self, width: int, height: int) -> StorageImage:
        vk = self._vk
        img_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=(
                vk.VK_IMAGE_USAGE_STORAGE_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            ),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        image = vk.vkCreateImage(self.device, img_info, None)
        reqs = vk.vkGetImageMemoryRequirements(self.device, image)
        alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=self._find_memory_type(
                reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            ),
        )
        memory = vk.vkAllocateMemory(self.device, alloc, None)
        vk.vkBindImageMemory(self.device, image, memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        view = vk.vkCreateImageView(self.device, view_info, None)
        return StorageImage(image=image, memory=memory, view=view, width=width, height=height)

    def _destroy_storage_image(self, img: Optional[StorageImage]) -> None:
        if img is None:
            return
        vk = self._vk
        vk.vkDestroyImageView(self.device, img.view, None)
        vk.vkDestroyImage(self.device, img.image, None)
        vk.vkFreeMemory(self.device, img.memory, None)

    def _begin_one_time_commands(self):
        vk = self._vk
        alloc = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.device, alloc)[0]
        begin = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin)
        return cmd

    def _submit_and_wait(self, cmd) -> None:
        vk = self._vk
        vk.vkEndCommandBuffer(cmd)
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(self.device, fence_info, None)
        vk.vkQueueSubmit(self.queue, 1, [submit], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10_000_000_000)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cmd])

    def _image_barrier(self, cmd, image, old_layout, new_layout, src_access, dst_access, src_stage, dst_stage):
        vk = self._vk
        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
            srcAccessMask=src_access,
            dstAccessMask=dst_access,
        )
        vk.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, None, 0, None, 1, [barrier])

    def _upload_rgba(self, image: StorageImage, pixels: np.ndarray) -> None:
        """Upload HxWx4 float32 pixels into a storage image."""
        vk = self._vk
        assert pixels.dtype == np.float32
        assert pixels.ndim == 3 and pixels.shape[2] == 4
        h, w, _ = pixels.shape
        if w != image.width or h != image.height:
            raise Exception(f"Upload size mismatch: {w}x{h} vs {image.width}x{image.height}")

        byte_size = pixels.nbytes
        staging, staging_mem = self._create_buffer(
            byte_size,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        ptr = vk.vkMapMemory(self.device, staging_mem, 0, byte_size, 0)
        vk.ffi.memmove(ptr, pixels.tobytes(), byte_size)
        vk.vkUnmapMemory(self.device, staging_mem)

        cmd = self._begin_one_time_commands()
        self._image_barrier(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_UNDEFINED,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )
        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=w, height=h, depth=1),
        )
        vk.vkCmdCopyBufferToImage(
            cmd,
            staging,
            image.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            [region],
        )
        self._image_barrier(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        )
        self._submit_and_wait(cmd)
        vk.vkDestroyBuffer(self.device, staging, None)
        vk.vkFreeMemory(self.device, staging_mem, None)

    def _download_rgba(self, image: StorageImage) -> np.ndarray:
        vk = self._vk
        w, h = image.width, image.height
        byte_size = w * h * 4 * 4
        staging, staging_mem = self._create_buffer(
            byte_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        cmd = self._begin_one_time_commands()
        self._image_barrier(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_TRANSFER_READ_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )
        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=w, height=h, depth=1),
        )
        vk.vkCmdCopyImageToBuffer(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            staging,
            1,
            [region],
        )
        self._submit_and_wait(cmd)

        ptr = vk.vkMapMemory(self.device, staging_mem, 0, byte_size, 0)
        # Some vulkan bindings return a cffi buffer directly from vkMapMemory
        if hasattr(ptr, "__buffer__") or type(ptr).__name__ == "buffer":
            raw = bytes(ptr)
        else:
            raw = bytes(vk.ffi.buffer(ptr, byte_size))
        arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(h, w, 4)
        vk.vkUnmapMemory(self.device, staging_mem)
        vk.vkDestroyBuffer(self.device, staging, None)
        vk.vkFreeMemory(self.device, staging_mem, None)
        return arr

    def _clear_image(self, image: StorageImage, color=(0.0, 0.0, 0.0, 1.0)) -> None:
        vk = self._vk
        cmd = self._begin_one_time_commands()
        self._image_barrier(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_UNDEFINED,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            0,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )
        vk.vkCmdClearColorImage(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VkClearColorValue(float32=list(color)),
            1,
            [
                vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                )
            ],
        )
        self._image_barrier(
            cmd,
            image.image,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        )
        self._submit_and_wait(cmd)

    def dispatch_rgba(
        self,
        pipeline: ComputePipeline,
        input_rgba: np.ndarray,
        push_constants: bytes,
        mask_rgba: Optional[np.ndarray] = None,
        workgroup: Tuple[int, int] = (16, 16),
    ) -> np.ndarray:
        """
        Run a compute shader on one HxWx4 float32 image.

        Bindings:
          0 = input (readonly)
          1 = output (writeonly)
          2 = mask (readonly; solid 1.0 alpha if unused)
        """
        vk = self._vk
        h, w, c = input_rgba.shape
        if c != 4:
            raise Exception(f"Expected HxWx4 image, got shape {input_rgba.shape}")

        input_img = self._create_storage_image(w, h)
        output_img = self._create_storage_image(w, h)
        mask_img = self._create_storage_image(w, h)
        try:
            self._upload_rgba(input_img, np.ascontiguousarray(input_rgba, dtype=np.float32))
            self._clear_image(output_img)
            if mask_rgba is None:
                mask_rgba = np.ones((h, w, 4), dtype=np.float32)
            else:
                mask_rgba = np.ascontiguousarray(mask_rgba, dtype=np.float32)
                if mask_rgba.ndim == 2:
                    mask_rgba = np.repeat(mask_rgba[..., None], 4, axis=2)
                elif mask_rgba.shape[-1] == 1:
                    mask_rgba = np.repeat(mask_rgba, 4, axis=2)
            self._upload_rgba(mask_img, mask_rgba)

            alloc = vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self.descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[pipeline.descriptor_set_layout],
            )
            descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc)[0]

            writes = []
            for binding, img in ((0, input_img), (1, output_img), (2, mask_img)):
                info = vk.VkDescriptorImageInfo(
                    sampler=vk.VK_NULL_HANDLE,
                    imageView=img.view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                )
                writes.append(
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set,
                        dstBinding=binding,
                        dstArrayElement=0,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        pImageInfo=[info],
                    )
                )
            vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

            cmd = self._begin_one_time_commands()
            # Ensure GENERAL layout after upload path
            for img in (input_img, output_img, mask_img):
                self._image_barrier(
                    cmd,
                    img.image,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                )

            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout,
                0,
                1,
                [descriptor_set],
                0,
                None,
            )
            if pipeline.push_constant_size > 0:
                data = push_constants
                if len(data) < pipeline.push_constant_size:
                    data = data + b"\x00" * (pipeline.push_constant_size - len(data))
                elif len(data) > pipeline.push_constant_size:
                    data = data[: pipeline.push_constant_size]
                cdata = vk.ffi.new("char[]", data)
                vk.vkCmdPushConstants(
                    cmd,
                    pipeline.pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    pipeline.push_constant_size,
                    cdata,
                )

            gx = int(math.ceil(w / float(workgroup[0])))
            gy = int(math.ceil(h / float(workgroup[1])))
            vk.vkCmdDispatch(cmd, gx, gy, 1)
            self._submit_and_wait(cmd)

            result = self._download_rgba(output_img)
            vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
            return result
        finally:
            self._destroy_storage_image(input_img)
            self._destroy_storage_image(output_img)
            self._destroy_storage_image(mask_img)

    def shutdown(self) -> None:
        if not self._initialized or self._vk is None:
            return
        vk = self._vk
        try:
            for pipe in self._pipeline_cache.values():
                vk.vkDestroyPipeline(self.device, pipe.pipeline, None)
                vk.vkDestroyPipelineLayout(self.device, pipe.pipeline_layout, None)
                vk.vkDestroyDescriptorSetLayout(self.device, pipe.descriptor_set_layout, None)
                vk.vkDestroyShaderModule(self.device, pipe.shader_module, None)
            self._pipeline_cache.clear()
            if self.descriptor_pool:
                vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            if self.command_pool:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            if self.device:
                vk.vkDestroyDevice(self.device, None)
            if self.instance:
                vk.vkDestroyInstance(self.instance, None)
        except Exception as e:
            logger.error("Vulkan shutdown error: %s", e)
        finally:
            self._initialized = False
            self.device = None
            self.instance = None
