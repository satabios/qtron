import os.path
import onnx
from data.private.config import private_models
from data.public.config import public_models
from onnx_tool import model_api_test

folder = public_models['folder']
test_set = {}
for modelinfo in public_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    detail = model_api_test(model, modelinfo['dynamic_input'])
    test_set[name] = detail
    print('-' * 64)

folder = private_models['folder']
for modelinfo in private_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    detail = model_api_test(model, modelinfo['dynamic_input'])
    test_set[name] = detail
    print('-' * 64)

from tabulate import tabulate

headers = ['model', 't_shapeinfer', 't_profile', 't_shapeinfer_v2', 't_profile_v2', 'Error']
rows = []
for key in test_set.keys():
    data = [key, test_set[key]['t_shapeinfer'], test_set[key]['t_profile']
        , test_set[key]['t_shapeinfer_new'], test_set[key]['t_profile_new']
        , test_set[key]['macsdiff']
            ]
    rows.append(data)
print(tabulate(rows, headers))
