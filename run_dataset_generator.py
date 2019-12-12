from models import create_model
from visualization import data_generator
from PyQt5 import QtWidgets

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    d = data_generator.DataGenerator()
    app.exec_()