# extracting semantic tags based on the DSG framework using open ai api
""" 
This will include 4 calls:
1. extract tuples from user description 2. extract questions from tupples 3. extract dependencies from tupples 4. only in the final prompt provide the image and not the description, and answer the questions in 2 based on it
"""

# demo before improving prompts:

#to review: tuple format doesn't align with semantic parts
#it is using the semantic parts we extracted before instead of looking at the whole text
TUPLE_EXTRACTION_PROMPT = """
SYSTEM:
You are generating Davidsonian Scene Graph (DSG) tuples from a text description.
Only include semantics explicitly stated in the text or in the provided semantic parts.
Do NOT add inferred objects/attributes/relations.

Return ONLY lines in this format:
id | tuple

Tuple format (use exactly one of these):
- ("entity", <object>)
- ("global", <global_feature>)
- ("attr:<category>", <value>, <object>)    where <category> ∈ {color,shape,size,material,texture,pose,action,state}
- ("rel:spatial", <relation>, <subject_object>, <object_object_or_scene>)
- ("rel:action", <relation>, <subject_object>, <object_object>)  (only if an action is explicitly between two objects)

Rules:
- Each tuple must be atomic (one fact).
- No duplicates.
- Use lowercase canonical object names.
- If “uncertainty” words appear (maybe/perhaps/seems), do NOT convert them to factual tuples; ignore them (they can be handled separately).

USER:
DESCRIPTION:
{DESCRIPTION_TEXT}

SEMANTIC PARTS (extracted lists; may be empty):
objects={OBJECTS}
attr_color={ATTR_COLOR}
attr_shape={ATTR_SHAPE}
attr_size={ATTR_SIZE}
attr_material={ATTR_MATERIAL}
attr_texture={ATTR_TEXTURE}
attr_pose={ATTR_POSE}
attr_action={ATTR_ACTION}
attr_state={ATTR_STATE}
spatial_relations={SPATIAL_RELATIONS}
world_knowledge={WORLD_KNOWLEDGE}
scene={SCENE}
camera_aspects={CAMERA_ASPECTS}
optical_effects={OPTICAL_EFFECTS}
subjective_detail={SUBJECTIVE_DETAIL}

Now output the DSG tuples.
"""

#they talk about negation handling but i don't think it makes sense according to paper:
# where does the id come from?

QUESTION_EXTRACTION_PROMPT = """
SYSTEM:
You are generating Davidsonian Scene Graph (DSG) tuples from a text description.
Only include semantics explicitly stated in the text or in the provided semantic parts.
Do NOT add inferred objects/attributes/relations.

Return ONLY lines in this format:
id | tuple

Tuple format (use exactly one of these):
- ("entity", <object>)
- ("global", <global_feature>)
- ("attr:<category>", <value>, <object>)    where <category> ∈ {color,shape,size,material,texture,pose,action,state}
- ("rel:spatial", <relation>, <subject_object>, <object_object_or_scene>)
- ("rel:action", <relation>, <subject_object>, <object_object>)  (only if an action is explicitly between two objects)

Rules:
- Each tuple must be atomic (one fact).
- No duplicates.
- Use lowercase canonical object names.
- If “uncertainty” words appear (maybe/perhaps/seems), do NOT convert them to factual tuples; ignore them (they can be handled separately).

USER:
DESCRIPTION:
{DESCRIPTION_TEXT}

SEMANTIC PARTS (extracted lists; may be empty):
objects={OBJECTS}
attr_color={ATTR_COLOR}
attr_shape={ATTR_SHAPE}
attr_size={ATTR_SIZE}
attr_material={ATTR_MATERIAL}
attr_texture={ATTR_TEXTURE}
attr_pose={ATTR_POSE}
attr_action={ATTR_ACTION}
attr_state={ATTR_STATE}
spatial_relations={SPATIAL_RELATIONS}
world_knowledge={WORLD_KNOWLEDGE}
scene={SCENE}
camera_aspects={CAMERA_ASPECTS}
optical_effects={OPTICAL_EFFECTS}
subjective_detail={SUBJECTIVE_DETAIL}

Now output the DSG tuples.
"""