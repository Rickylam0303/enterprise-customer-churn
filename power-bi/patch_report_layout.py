"""Patch captions and the actual segment visual filter; never change the embedded model.

The output requires Power BI Desktop open/refresh validation. Keep the input as a
backup. Canonical M/DAX source must still be applied to refresh the new holdout data.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def patch_layout(layout, minimum=1000):
    changed = {"kpi_captions": 0, "segment_filters": 0}
    for page in layout["sections"]:
        for visual in page.get("visualContainers", []):
            config = json.loads(visual["config"])
            single = config.get("singleVisual", {})
            query = json.loads(visual["query"]) if visual.get("query") else None
            if not query: continue
            semantic = query["Commands"][0]["SemanticQueryDataShapeCommand"]["Query"]
            measures = {s.get("Measure", {}).get("Property") for s in semantic.get("Select", [])}
            caption = None
            if single.get("visualType") == "cardVisual":
                if "At Risk Count" in measures: caption = "Historical Churned Customers"
                elif "ML At Risk Count" in measures: caption = "Predicted High-Risk Customers"
                if caption:
                    for q in [semantic, single.get("prototypeQuery", {})]:
                        for selected in q.get("Select", []): selected["NativeReferenceName"] = caption
                    single.setdefault("vcObjects", {})["title"] = [{"properties": {
                        "show": {"expr": {"Literal": {"Value": "true"}}},
                        "text": {"expr": {"Literal": {"Value": "'" + caption + "'"}}},
                    }}]
                    changed["kpi_captions"] += 1
            if single.get("visualType") == "tableEx" and "Segment Customer Count" in measures and "Segment Churn Rate" in measures:
                alias = semantic["From"][0]["Name"]
                condition = {"Condition": {"Comparison": {"ComparisonKind": 2,
                    "Left": {"Measure": {"Expression": {"SourceRef": {"Source": alias}}, "Property": "Segment Customer Count"}},
                    "Right": {"Literal": {"Value": f"{minimum}L"}}}}}
                # Persist in both the executable query and the visual's prototype.
                for q in [semantic, single["prototypeQuery"]]:
                    q.setdefault("Where", []).append(copy.deepcopy(condition))
                filters = json.loads(visual.get("filters", "[]"))
                filters.append({"name": "portfolio_min_segment_count", "type": "Advanced", "howCreated": 1,
                    "expression": {"Measure": {"Expression": {"SourceRef": {"Entity": "train"}}, "Property": "Segment Customer Count"}},
                    "filter": {"Version": 2, "From": copy.deepcopy(semantic["From"]), "Where": [condition]},
                    "isHiddenInViewMode": False})
                visual["filters"] = json.dumps(filters, separators=(",", ":"))
                changed["segment_filters"] += 1
            visual["config"] = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
            visual["query"] = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    if changed != {"kpi_captions": 2, "segment_filters": 1}:
        raise ValueError(f"Unexpected report bindings; do not write a guessed report: {changed}")
    return changed


def patch_report(source, destination, minimum=1000):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination: raise ValueError("Use a separate output and retain the source backup.")
    if destination.exists(): raise FileExistsError(destination)
    with zipfile.ZipFile(source) as before:
        layout = json.loads(before.read("Report/Layout").decode("utf-16le"))
        changes = patch_layout(layout, minimum)
        with zipfile.ZipFile(destination, "w") as after:
            for entry in before.infolist():
                if entry.filename == "SecurityBindings": continue
                data = before.read(entry.filename)
                if entry.filename == "Report/Layout":
                    data = json.dumps(layout, ensure_ascii=False, separators=(",", ":")).encode("utf-16le")
                elif entry.filename == "[Content_Types].xml":
                    tree = ET.fromstring(data)
                    for child in list(tree):
                        if "SecurityBindings" in child.attrib.get("PartName", ""): tree.remove(child)
                    ET.register_namespace("", "http://schemas.openxmlformats.org/package/2006/content-types")
                    data = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
                after.writestr(entry, data)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(destination) as after:
        if after.testzip() is not None or before.read("DataModel") != after.read("DataModel"):
            raise ValueError("Invalid package or embedded model changed.")
        json.loads(after.read("Report/Layout").decode("utf-16le"))
    return {**changes, "minimum": minimum, "data_model_unchanged": True, "desktop_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source"); parser.add_argument("destination"); parser.add_argument("--minimum", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(patch_report(args.source, args.destination, args.minimum), indent=2))
