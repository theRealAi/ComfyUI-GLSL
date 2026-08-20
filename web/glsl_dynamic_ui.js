import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PROCESSOR = "GLSLGPUProcessor";
const CORE_WIDGETS = new Set(["shader", "enabled"]);
const CORE_INPUTS = new Set(["image", "mask"]);

async function fetchShaderSchema(path) {
  const res = await api.fetchApi(
    `/glsl/shader_metadata?path=${encodeURIComponent(path)}`
  );
  if (!res.ok) {
    return null;
  }
  return await res.json();
}

function setWidgetVisible(node, widget, visible) {
  if (!widget) return;
  widget.hidden = !visible;
  // Collapse hidden widgets so the node doesn't stay huge
  if (visible) {
    delete widget.computeSize;
  } else {
    widget.computeSize = () => [0, -4];
  }
  if (widget.element) {
    widget.element.style.display = visible ? "" : "none";
  }
}

function refreshNodeSize(node) {
  const computed = node.computeSize?.() || node.size;
  const width = Math.max(node.size?.[0] || 210, computed[0] || 210);
  const height = computed[1] || node.size?.[1] || 100;
  node.setSize?.([width, height]);
  node.setDirtyCanvas?.(true, true);
}

function applyProcessorSchema(node, schema) {
  if (!schema) return;
  const uniformNames = new Set(schema.uniform_names || []);
  const inputNames = new Set(schema.input_names || []);

  for (const widget of node.widgets || []) {
    if (CORE_WIDGETS.has(widget.name)) {
      setWidgetVisible(node, widget, true);
      continue;
    }
    setWidgetVisible(node, widget, uniformNames.has(widget.name));
  }

  for (const input of node.inputs || []) {
    if (!input || CORE_INPUTS.has(input.name)) {
      continue;
    }
    const wanted =
      inputNames.has(input.name) ||
      (input.name === "mask" && inputNames.has("mask"));
    input.hidden = !wanted;
  }

  if (schema.description) {
    node.title = `GLSL GPU Processor`;
    node.properties = node.properties || {};
    node.properties.glsl_shader_name = schema.name || "";
    node.properties.glsl_shader_desc = schema.description || "";
  }

  refreshNodeSize(node);
}

function hookShaderWidget(node) {
  const shaderWidget = node.widgets?.find((w) => w.name === "shader");
  if (!shaderWidget || shaderWidget._glslHooked) {
    return;
  }
  shaderWidget._glslHooked = true;
  const prev = shaderWidget.callback;
  shaderWidget.callback = async function (value) {
    if (prev) prev.call(this, value);
    const schema = await fetchShaderSchema(value);
    applyProcessorSchema(node, schema);
  };
  if (shaderWidget.value) {
    fetchShaderSchema(shaderWidget.value).then((schema) =>
      applyProcessorSchema(node, schema)
    );
  }
}

app.registerExtension({
  name: "ComfyUI-GLSL.DynamicShaderUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== PROCESSOR) {
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      // Hide all non-core widgets until schema arrives (avoids huge first paint)
      for (const widget of this.widgets || []) {
        if (!CORE_WIDGETS.has(widget.name)) {
          setWidgetVisible(this, widget, false);
        }
      }
      hookShaderWidget(this);
      refreshNodeSize(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      hookShaderWidget(this);
      const shaderWidget = this.widgets?.find((w) => w.name === "shader");
      if (shaderWidget?.value) {
        fetchShaderSchema(shaderWidget.value).then((schema) =>
          applyProcessorSchema(this, schema)
        );
      } else {
        refreshNodeSize(this);
      }
      return r;
    };
  },
});
