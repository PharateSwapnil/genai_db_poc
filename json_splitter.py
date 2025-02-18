import json
import copy
import requests
from langchain_text_splitters import RecursiveJsonSplitter

# This is a large nested json object and will be loaded as a python dict
json_data = None # requests.get("https://api.smith.langchain.com/openapi.json").json()
json_data_path = "/home/richhiey/Desktop/code/genai/data/conventional_power_plants/datapackage.json"
with open(json_data_path, "r") as f:
    json_data = json.load(f)

splitter = RecursiveJsonSplitter(max_chunk_size=300)
# Recursively split json data - If you need to access/manipulate the smaller json chunks
# json_chunks = splitter.split_json(json_data=json_data)

def split_json_custom(json_data):
    resources =  json_data["resources"]
    filtered_resources = []
    for resource in resources:
        if resource.get("schema") and resource.get("profile") == "tabular-data-resource":
            resource_schema = resource["schema"]
            filtered_resource = {}
            if resource.get("title"):
                filtered_resource["table_description"] = resource["title"]
            if resource.get("name"):
                filtered_resource["table_name"] = resource["name"]
            resource_schema_fields = resource_schema["fields"]
            filtered_resource["primary_key"] = resource_schema["primaryKey"]
            for field in resource_schema_fields:
                filt_resource_copy = copy.deepcopy(filtered_resource)
                field_resource = filt_resource_copy | field
                field_resource["column_name"] = field_resource.pop("name")
                filtered_resources.append(field_resource)
    return filtered_resources

json_chunks = split_json_custom(json_data=json_data)