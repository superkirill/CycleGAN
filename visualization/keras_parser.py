class KerasParser():
    """Extract separate layers from Keras nn"""
    def __init__(self, model):
        self.model = model
        
    def get_layers(self):
        return []