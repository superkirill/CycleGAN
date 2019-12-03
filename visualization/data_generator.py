from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import os
import cv2
import numpy as np


class DataGenerator(QtWidgets.QMainWindow):
    """Interface for creating toy colorization datasets"""

    def __init__(self):
        super(DataGenerator, self).__init__()
        # Load user interface design
        uic.loadUi('visualization/create_datasets_design.ui', self)
        # Connect signals with slots
        self.checkBox_add_triangles.stateChanged.connect(self.check_triangles_checked)
        self.checkBox_add_circles.stateChanged.connect(self.check_circles_checked)
        self.radioButton_triangles.toggled.connect(self.radio_button_triangles_checked)
        self.radioButton_circles.toggled.connect(self.radio_button_circles_checked)
        self.radioButton_red.toggled.connect(self.radio_button_red_checked)
        self.radioButton_green.toggled.connect(self.radio_button_green_checked)
        self.radioButton_blue.toggled.connect(self.radio_button_blue_checked)
        self.checkBox_add_noise.stateChanged.connect(self.add_noise)
        self.lineEdit.textChanged.connect(self.update_sigma)
        self.checkBox_center.stateChanged.connect(self.center_figure)
        self.pushButton_generate.pressed.connect(self.save)
        self.colors = {'blue':(0,0,255), 'red':(254,0,0), 'green':(0,130,0)}
        self.root = "D:/Документы/University of Bordeaux/TRDP/Code/Datasets"
        # Currently displaying object
        self.to_draw = "triangles"
        # Colors of the objects
        self.triangles_color = self.colors['red']
        self.circles_color = self.colors['red']
        # Generation parameters
        self.triangles_center = True
        self.triangles_noisy = False
        self.triangles_sigma = 120
        self.circles_center = True
        self.circles_noisy = False
        self.circles_sigma = 120
        # Set initial cursor position in the sigma lineEdit
        self.lineEdit.setCursorPosition(2)
        self.show()

    def check_triangles_checked(self, state):
        """Handle a mouse click on the QCheckBox, that changes the
        state of the checkbox, by enabling editing of the triangles'
        generation parameters

            Parameters:
                state - Qt.Checked or Qt.Unchecked
        """
        if state == Qt.Checked:
            self.lineEdit_triangles_train.setEnabled(True)
            self.lineEdit_triangles_test.setEnabled(True)
            self.radioButton_triangles.setEnabled(True)
            self.checkBox_add_noise.setEnabled(True)
            self.checkBox_center.setEnabled(True)
            self.radioButton_red.setEnabled(True)
            self.radioButton_green.setEnabled(True)
            self.radioButton_blue.setEnabled(True)
            self.radioButton_triangles.setChecked(True)
        elif state == Qt.Unchecked:
            self.lineEdit_triangles_train.setEnabled(False)
            self.lineEdit_triangles_test.setEnabled(False)
            self.radioButton_triangles.setEnabled(False)
            self.checkBox_add_noise.setEnabled(False)
            self.checkBox_center.setEnabled(False)
            self.radioButton_red.setEnabled(False)
            self.radioButton_green.setEnabled(False)
            self.radioButton_blue.setEnabled(False)
            self.radioButton_triangles.setChecked(False)
        self.pushButton_generate.setEnabled(True)
    
    def check_circles_checked(self, state):
        """Handle a mouse click on the QCheckBox, that changes the
        state of the checkbox, by enabling editing of the circles'
        generation parameters

            Parameters:
                state - Qt.Checked or Qt.Unchecked
        """
        if state == Qt.Checked:
            self.lineEdit_circles_train.setEnabled(True)
            self.lineEdit_circles_test.setEnabled(True)
            self.radioButton_circles.setEnabled(True)
            self.checkBox_add_noise.setEnabled(True)
            self.checkBox_center.setEnabled(True)
            self.radioButton_red.setEnabled(True)
            self.radioButton_green.setEnabled(True)
            self.radioButton_blue.setEnabled(True)
            self.radioButton_circles.setChecked(True)
        elif state == Qt.Unchecked:
            self.lineEdit_circles_train.setEnabled(False)
            self.lineEdit_circles_test.setEnabled(False)
            self.radioButton_circles.setEnabled(False)
            self.checkBox_add_noise.setEnabled(False)
            self.checkBox_center.setEnabled(False)
            self.radioButton_red.setEnabled(False)
            self.radioButton_green.setEnabled(False)
            self.radioButton_blue.setEnabled(False)
            self.radioButton_circles.setChecked(False)

    def add_noise(self, state):
        """Handle a mouse click on the QCheckBox, that changes the
        state of the checkbox, by enabling addition of the gaussian
        noise to the generated images

            Parameters:
                state - Qt.Checked or Qt.Unchecked
        """
        if state == Qt.Checked:
            if self.to_draw == "triangles":
                self.triangles_noisy = True
            else:
                self.circles_noisy= True
            self.lineEdit.setEnabled(True)
        else:
            if self.to_draw == "triangles":
                self.triangles_noisy = False
            else:
                self.circles_noisy= False
            self.lineEdit.setEnabled(False)
        self.draw()

    def radio_button_triangles_checked(self, checked):
        """Handle a mouse click on the QRadioButton, that changes the
        state of the radiobutton, by accessing the parameters of triangles'
        generation process

            Parameters:
                checked - True or False
        """
        if checked is True:
            self.to_draw = "triangles"
            self.checkBox_add_noise.setChecked(self.triangles_noisy)
            self.lineEdit.setText(str(self.triangles_sigma))
            self.checkBox_center.setChecked(self.triangles_center)
            if self.triangles_color == self.colors['red']:
                self.radioButton_red.toggle()
            elif self.triangles_color == self.colors['green']:
                self.radioButton_green.toggle()
            else:
                self.radioButton_blue.toggle()
            self.draw()

    def radio_button_circles_checked(self, checked):
        """Handle a mouse click on the QRadioButton, that changes the
        state of the radiobutton, by accessing the parameters of circles'
        generation process

            Parameters:
                checked - True or False
        """
        if checked is True:
            self.to_draw = "circles"
            self.checkBox_add_noise.setChecked(self.circles_noisy)
            self.lineEdit.setText(str(self.circles_sigma))
            self.checkBox_center.setChecked(self.circles_center)
            if self.circles_color == self.colors['red']:
                self.radioButton_red.toggle()
            elif self.circles_color == self.colors['green']:
                self.radioButton_green.toggle()
            else:
                self.radioButton_blue.toggle()
            self.draw()

    def radio_button_red_checked(self, checked):
        """Handle a mouse click on the QRadioButton, that changes the
        state of the radiobutton, by setting the color of currently
        selected objects to be red

            Parameters:
                checked - True or False
        """
        if checked is True:
            if self.to_draw == "triangles":
                self.triangles_color = self.colors['red']
            else:
                self.circles_color = self.colors['red']
            self.draw()

    def radio_button_green_checked(self, checked):
        """Handle a mouse click on the QRadioButton, that changes the
        state of the radiobutton, by setting the color of currently
        selected objects to be green

            Parameters:
                checked - True or False
        """
        if checked is True:
            if self.to_draw == "triangles":
                self.triangles_color = self.colors['green']
            else:
                self.circles_color = self.colors['green']
            self.draw()

    def radio_button_blue_checked(self, checked):
        """Handle a mouse click on the QRadioButton, that changes the
        state of the radiobutton, by setting the color of currently
        selected objects to be blue

            Parameters:
                checked - True or False
        """
        if checked is True:
            if self.to_draw == "triangles":
                self.triangles_color = self.colors['blue']
            else:
                self.circles_color = self.colors['blue']
            self.draw()

    def update_sigma(self, text):
        """Handles the textChanged signal emitted by the lineEdit
        containing the sigma value

            Parameters:
                text - a string containing the text of the lineEdit
        """
        if len(text) > 1 and text[0] == "0":
            while text[0] == "0" and len(text) > 1:
                text = text[1:]
            pos = self.lineEdit.cursorPosition()
            self.lineEdit.setText(text)
            self.lineEdit.setCursorPosition(pos-1)
        elif text == "":
            self.lineEdit.setText("0")
            if self.to_draw == "triangles":
                self.triangles_sigma = 0
            else:
                self.circles_sigma = 0
            self.lineEdit.setCursorPosition(0)
        else:
            if self.to_draw == "triangles":
                self.triangles_sigma = int(text)
            else:
                self.circles_sigma = int(text)

        self.draw()

    def center_figure(self, state):
        """Handle a mouse click on the QCheckBox, that changes the
        state of the checkbox, by positioning generated objects in
        the middle of the image

            Parameters:
                state - Qt.Checked or Qt.Unchecked
        """
        if state == Qt.Checked:
            if self.to_draw == "triangles":
                self.triangles_center = True
            else:
                self.circles_center = True
        else:
            if self.to_draw == "triangles":
                self.triangles_center = False
            else:
                self.circles_center = False
        self.draw()

    def gen_triang(self, canvas=(256, 256), color=(0, 255, 0), center=True):
        """Generate a triangle

            Parameters:
                canvas - a tuple (w,h), representing the size of the image
                color - a tuple (R, G, B), representing the color of the triangle
                center - boolean, indicating whether the triangle is to be positioned
                    in the center of the image
            Return value:
                numpy array representing the image of the triangle
        """
        w, h = canvas[0], canvas[1]
        image = np.zeros((w, h, 3), np.uint8) * 255
        pt1 = ((np.random.rand()) * w, (np.random.rand()) * h)
        pt2 = ((np.random.rand()) * w, (np.random.rand()) * h)
        pt3 = ((np.random.rand()) * w, (np.random.rand()) * h)
        triangle_cnt = np.array([pt1, pt2, pt3]).reshape((-1, 1, 2)).astype(np.int32)
        if center is True:
            center = (np.sum(triangle_cnt, 0) / 3)
            shift = np.array([w / 2, h / 2]) - center
            triangle_cnt += shift.astype(np.int32)
        cv2.drawContours(image, [triangle_cnt], 0, color, -1)
        if self.triangles_noisy is True:
            image = self.add_gaussian_noise(image, self.triangles_sigma)
        return image

    def gen_circle(self, canvas=(256, 256), color=(0, 255, 0), center=True):
        """Generate a circle

            Parameters:
                canvas - a tuple (w,h), representing the size of the image
                color - a tuple (R, G, B), representing the color of the circle
                center - boolean, indicating whether the circle is to be positioned
                    in the center of the image
            Return value:
                numpy array representing the image of the circle
        """
        w, h = canvas[0], canvas[1]
        image = np.zeros((w, h, 3), np.uint8) * 255
        radius = np.random.rand() * min(w, h) / 3
        if center is True:
            xc = w / 2
            yc = h / 2
        else:
            xc = np.random.rand() * w
            yc = np.random.rand() * h
            xc = radius if xc - radius < 0 else xc
            xc = w - radius if xc + radius > w else xc
            yc = radius if yc - radius < 0 else yc
            yc = h - radius if yc + radius > h else yc
        cv2.circle(image, (int(xc), int(yc)), int(radius), color, -1)
        if self.circles_noisy is True:
            image = self.add_gaussian_noise(image, self.circles_sigma)
        return image

    def add_gaussian_noise(self, ima, sigma=120):
        """Add gaussian noise to the image with a given sigma

            Parameters:
                ima - the image to which noise is to be added
                sigma - sigma of the gaussian distribution
            Return value:
                numpy array representing the noisy image
        """
        rows, columns, channels = ima.shape
        noise = np.zeros((rows, columns, channels))
        mean = (0, 0, 0)
        s = (sigma, sigma, sigma)
        cv2.randn(noise, mean, s)
        noisy_image = np.maximum(np.minimum((ima + noise), 255), 0)
        return noisy_image.astype(np.uint8)

    def save(self):
        """Handle the clicked signal of the pushButton_save widget by
            generating and saving the objects to the specified directory
        """
        if self.checkBox_add_triangles.checkState() == Qt.Checked:
            t_train = int(self.lineEdit_triangles_train.text())
            t_test = int(self.lineEdit_triangles_test.text())
        if self.checkBox_add_circles.checkState() == Qt.Checked:
            c_train = int(self.lineEdit_circles_train.text())
            c_test = int(self.lineEdit_circles_test.text())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory",
                                                          self.root,
                                                          QtWidgets.QFileDialog.ShowDirsOnly
                                                          )
        os.mkdir(path + '/testA')
        os.mkdir(path + '/testB')
        os.mkdir(path + '/trainA')
        os.mkdir(path + '/trainB')
        if self.checkBox_add_triangles.checkState() == Qt.Checked:
            for i in range(t_train):
                image = self.gen_triang(color=tuple(list(self.triangles_color)[::-1]), center=self.triangles_center)
                cv2.imwrite(path + "/trainA/t%d.jpg" % i, image)
                cv2.imwrite(path + "/trainB/t%d.jpg" % i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            for i in range(t_test):
                image = self.gen_triang(color=tuple(list(self.triangles_color)[::-1]), center=self.triangles_center)
                cv2.imwrite(path + "/testA/t%d.jpg" % i, image)
                cv2.imwrite(path + "/testB/t%d.jpg" % i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if self.checkBox_add_circles.checkState() == Qt.Checked:
            for i in range(c_train):
                image = self.gen_circle(color=tuple(list(self.circles_color)[::-1]), center=self.circles_center)
                cv2.imwrite(path + "/trainA/c%d.jpg" % i, image)
                cv2.imwrite(path + "/trainB/c%d.jpg" % i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            for i in range(c_test):
                image = self.gen_circle(color=tuple(list(self.circles_color)[::-1]), center=self.circles_center)
                cv2.imwrite(path + "/testA/c%d.jpg" % i, image)
                cv2.imwrite(path + "/testB/c%d.jpg" % i, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def draw(self):
        """Draw an example of the generated object in the QGraphicsView widget"""
        if self.to_draw == "triangles":
            image = self.gen_triang(color=self.triangles_color, center=self.triangles_center)
        else:
            image = self.gen_circle(color=self.circles_color, center=self.circles_center)
        height, width, channel = image.shape
        image_format = QImage.Format_RGB888
        bytes_per_line = 3 * width
        view = self.graphicsView
        scene = QtWidgets.QGraphicsScene()
        view.setScene(scene)
        qt_image = QImage(image.tobytes(), width, height, bytes_per_line, image_format)
        scene.addPixmap(QPixmap(qt_image))