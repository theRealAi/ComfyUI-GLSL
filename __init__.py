# ComfyUI-GLSL Package Entry Point

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _register_api_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    from .src.shader.ui_schema import list_shader_relpaths, shader_ui_schema

    routes = PromptServer.instance.routes

    @routes.get("/glsl/shaders")
    async def glsl_list_shaders(request):
        return web.json_response({"shaders": list_shader_relpaths()})

    @routes.get("/glsl/shader_metadata")
    async def glsl_shader_metadata(request):
        path = request.rel_url.query.get("path", "")
        if not path or ".." in path:
            return web.json_response({"error": "invalid path"}, status=400)
        try:
            return web.json_response(shader_ui_schema(path))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)


_register_api_routes()
