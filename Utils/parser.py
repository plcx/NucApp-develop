import yaml
import ast

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def parse_tuple(string):
    s = ast.literal_eval(str(string))
    if type(s) == tuple:
        return s
    return
