#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:50:54 2021

@author: siva, derro
"""

from os.path import exists
""" import tflite_runtime.interpreter as tflite """
import onnxruntime as ort 
import tensorflow as tf

""" class tflitebase: # base class that provides the framework for tflite model loading and prediction

    def __init__(self, modelfile):
        self.ready = False
        assert exists(modelfile)
        self.interpreter = tflite.Interpreter(model_path=modelfile)
        self.interpreter.allocate_tensors()
        self.input_indices = [i["index"] for i in self.interpreter.get_input_details()]
        self.output_indices = [o["index"] for o in self.interpreter.get_output_details()]
        self.ready = True
        return

    def predict(self, x):
        assert self.ready
        if not isinstance(x, list): x = [x]
        assert len(x) == len(self.input_indices)
        for i, e in zip(self.input_indices, x):
            self.interpreter.set_tensor(i, e)
        self.interpreter.invoke()
        out = [self.interpreter.get_tensor(i) for i in self.output_indices]
        if len(out) == 1: out = out[0]
        return out """


class onnxbase:

    def __init__(self, modelfile):
        self.ready = False
        assert exists(modelfile)
        self.sess = ort.InferenceSession(modelfile, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.ready = True
        return

    def cuda(self):
        self.sess.set_providers(["CUDAExecutionProvider"])
        return

    def predict(self, x):
        assert self.ready
        if not isinstance(x, list): x = [x]
        assert len(x) == len(self.input_names)
        x = {n: e for n, e in zip(self.input_names, x)}
        out = self.sess.run(self.output_names, input_feed=x)
        if len(out) == 1: out = out[0]
        return out

class kerasbase:
    def __init__(self, modelfile):
        self.ready = False
        assert exists(modelfile)
        # Load the model
        self.model = tf.keras.models.load_model(modelfile, compile=False)
        # Get input and output names
        self.input_names = [input_layer.name for input_layer in self.model.inputs]
        self.output_names = [output_layer.name for output_layer in self.model.outputs]
        self.ready = True
    
    def predict(self, x):
        assert self.ready
        # Check if input is a list, if not convert it into a list
        if not isinstance(x, list):
            x = [x]
        # Check if number of inputs matches the number of input names
        assert len(x) == len(self.input_names)
        # Prepare input dictionary
        input_dict = {input_name: input_data for input_name, input_data in zip(self.input_names, x)}
        # Perform prediction
        out = self.model.predict(input_dict)
        # If there is only one output, return it directly
        if len(out) == 1: out = out[0]
        return out
    