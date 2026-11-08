#!/usr/bin/env python3
"""Convert a UI workflow copy; enable materialization on the chosen final pair only."""

import argparse
import copy
import json
from pathlib import Path

TYPES = {"ModelMergeSimple": "ModelMergeSimpleRelease", "CLIPMergeSimple": "CLIPMergeSimpleRelease"}


def validate_graph(workflow):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    if len(nodes) != len(workflow["nodes"]):
        raise ValueError("Duplicate node IDs")
    links = {link[0]: link for link in workflow["links"]}
    if len(links) != len(workflow["links"]):
        raise ValueError("Duplicate link IDs")
    for link_id, source, source_slot, target, target_slot, kind in links.values():
        output = nodes[source]["outputs"][source_slot]
        input_ = nodes[target]["inputs"][target_slot]
        if link_id not in (output.get("links") or []) or input_.get("link") != link_id:
            raise ValueError(f"Link {link_id} is inconsistent with its sockets")
        if kind != "*" and any(socket["type"] not in (kind, "*") for socket in (output, input_)):
            raise ValueError(f"Link {link_id} has incompatible socket types")
    for node in nodes.values():
        for index, input_ in enumerate(node.get("inputs", [])):
            link_id = input_.get("link")
            if link_id is not None and (link_id not in links or links[link_id][3:5] != [node["id"], index]):
                raise ValueError(f"Dangling input link {link_id}")
        for index, output in enumerate(node.get("outputs", [])):
            for link_id in output.get("links") or []:
                if link_id not in links or links[link_id][1:3] != [node["id"], index]:
                    raise ValueError(f"Dangling output link {link_id}")


def convert(workflow, final_model=2023, final_clip=2022):
    validate_graph(workflow)
    result = copy.deepcopy(workflow)
    found = {}
    for node in result["nodes"]:
        old_type = node["type"]
        if old_type not in TYPES and old_type not in TYPES.values():
            continue
        new_type = TYPES.get(old_type, old_type)
        enabled = node["id"] in (final_model, final_clip)
        values = node["widgets_values"]
        if not isinstance(values, list) or len(values) != (1 if old_type in TYPES else 2):
            raise ValueError(f"Unexpected widget layout at node {node['id']}")
        if len(values) == 1:
            values.append(enabled)
        else:
            values[1] = enabled
        if "widgets_values_named" in node:
            node["widgets_values_named"]["release_sources"] = enabled
        if not any(i["name"] == "release_sources" for i in node["inputs"]):
            node["inputs"].append({"localized_name": "release_sources", "name": "release_sources",
                                   "type": "BOOLEAN", "widget": {"name": "release_sources"}, "link": None})
        if "size" in node:
            node["size"][1] = max(node["size"][1], 102)
        node["type"] = new_type
        properties = node.setdefault("properties", {})
        properties["Node name for S&R"] = new_type
        # Do not advertise these custom classes as nodes shipped by comfy-core.
        properties.pop("cnr_id", None)
        properties.pop("ver", None)
        if node["id"] in (final_model, final_clip):
            found[node["id"]] = new_type
    if found != {final_model: "ModelMergeSimpleRelease", final_clip: "CLIPMergeSimpleRelease"}:
        raise ValueError("Final MODEL/CLIP IDs do not identify the expected merge nodes")
    validate_graph(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-model", type=int, default=2023)
    parser.add_argument("--final-clip", type=int, default=2022)
    args = parser.parse_args()
    output = args.output or args.source.with_name(args.source.stem + "_ReleaseSources.json")
    if args.source.resolve() == output.resolve():
        parser.error("The output must be a new file, never the source workflow")
    converted = convert(json.loads(args.source.read_text()), args.final_model, args.final_clip)
    # Exclusive creation prevents accidental overwrite of another workflow.
    with output.open("x") as stream:
        json.dump(converted, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
