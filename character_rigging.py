# src/character_rigging.py

import bpy
import sys
import os

def import_character_model(model_path):
    """
    Import a rigged character model into Blender.
    """
    bpy.ops.import_scene.fbx(filepath=model_path)
    # Adjust import settings as needed

def apply_texture(character_object_name, texture_image_path):
    """
    Apply the source image as a texture to the character model.
    """
    mat = bpy.data.materials.new(name="SourceTexture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_image_path)
    
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    character = bpy.data.objects.get(character_object_name)
    if not character:
        raise ValueError(f"Character object '{character_object_name}' not found in the scene.")
    
    if character.data.materials:
        character.data.materials[0] = mat
    else:
        character.data.materials.append(mat)

def setup_scene(model_path, texture_image_path, character_object_name):
    """
    Set up the Blender scene with the character model and texture.
    """
    import_character_model(model_path)
    apply_texture(character_object_name, texture_image_path)
    # Additional setup like lighting, camera, etc., can be added here
