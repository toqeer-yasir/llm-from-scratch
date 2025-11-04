import os
import json
import numpy as np
import tensorflow as tf

def gpt2_settings_and_params(model_path):

    settings_path = os.path.join(model_path, "hparams.json")
    settings = json.load(open(settings_path, "r", encoding="utf-8"))
    
    tf_ckpt_path = tf.train.latest_checkpoint(model_path)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    
    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]
        
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params