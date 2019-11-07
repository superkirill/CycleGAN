from . import pytorch_parser, keras_parser

class Parser():
    """Extracts separate layers from different nn frameworks"""
    def __init__(self, framework, model):
        if framework == "PyTorch":
            self.framework_parser = pytorch_parser.PyTorchParser(model)
        elif framework == "Keras":
            self.framework_parser = keras_parser.KerasParser(model)
        else:
            raise Exception("Framework is not supported")

    def get_layers(self):
        return self.framework_parser.get_layers()

