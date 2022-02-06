# PyQt image editor template

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QKeySequence, QPixmap, QImage

import numpy as np
from PIL import Image

import sys
import json


class Slider:

    def __init__(self, label: str, default: float, on_update):
        self.on_update = on_update
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(100*default+0.5))
        self.slider.valueChanged.connect(self._on_update)
        self.label_text = label
        self.label = QLabel(label)
        self.label.setBuddy(self.slider)
        self.update_label_text()

    def add_to_layout(self, layout):
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

    def update_label_text(self):
        """Displays integer value between -50 and 50"""
        value = self.slider.value()
        self.label.setText(self.label_text+" ({:+d})".format(value-50))

    def get_value(self) -> float:
        """Returns float value between 0 and 1"""
        return 0.01*self.slider.value()

    def set_value(self, value: float):
        """Receives float value between 0 and 1"""
        value_i = min(max(int(round(100*value)), 0), 100)
        self.slider.setValue(value_i)

    def _on_update(self):
        self.on_update()
        self.update_label_text()


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        self.setWindowTitle("PyQt image editor template")

        # state
        self.state = {
            'imagePath': "",
            'exposure': None,
            'gamma': None,
            'hueshift': None
        }

        # menu bar
        self.fileMenu = self.menuBar().addMenu("&File")
        self.openImageAction = QAction("&Open image")
        self.openImageAction.triggered.connect(self.openImage)
        self.openImageAction.setShortcut(QKeySequence.Open)
        self.fileMenu.addAction(self.openImageAction)
        self.saveImageAction = QAction("&Save image")
        self.saveImageAction.triggered.connect(self.saveImage)
        self.saveImageAction.setShortcut(QKeySequence.Save)
        self.fileMenu.addAction(self.saveImageAction)
        self.openSettingsAction = QAction("&Open settings")
        self.openSettingsAction.triggered.connect(self.openSettings)
        self.fileMenu.addAction(self.openSettingsAction)
        self.saveSettingsAction = QAction("&Save settings")
        self.saveSettingsAction.triggered.connect(self.saveSettings)
        self.fileMenu.addAction(self.saveSettingsAction)
        self.editMenu = self.menuBar().addMenu("&Edit")

        # adjustment sliders
        self.exposureSlider = Slider("Exposure", 0.5,
                                     self.updateImageAdjustments)
        self.gammaSlider = Slider("Gamma", 0.5,
                                  self.updateImageAdjustments)
        self.hueshiftSlider = Slider("Hue shift", 0.5,
                                     self.updateImageAdjustments)
        self.updateState()

        # image
        self.imageContainer = None
        self.imageOriginal = self.openImageFilename(self.state['imagePath'])
        self.imageScaled = self.scaleImage(self.imageOriginal)
        self.imageAdjusted = self.applyAdjustments(self.imageScaled)
        self.updateDisplayImage(self.imageAdjusted, reload=True)

        self.updateWindow()

    def updateWindow(self):
        """init/update layouts/widgets in the window"""
        self.controlBar = QVBoxLayout()
        self.exposureSlider.add_to_layout(self.controlBar)
        self.gammaSlider.add_to_layout(self.controlBar)
        self.hueshiftSlider.add_to_layout(self.controlBar)
        self.controlBar.addStretch(1)
        self.controlBarContainer = QWidget()
        self.controlBarContainer.setLayout(self.controlBar)
        self.controlBarContainer.setMaximumWidth(300)
        self.controlBarContainer.setMinimumWidth(180)

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.controlBarContainer, 0, 0)
        self.mainLayout.addWidget(self.imageContainer, 0, 1)
        self.mainLayout.setRowStretch(1, 1)
        #self.mainLayout.setColumnStretch(0, 1)
        #self.mainLayout.setColumnStretch(1, 1)

        self.centralWidget = QWidget(self)
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

    def openImage(self):
        """Call this function to open an image file"""
        filepath = QFileDialog.getOpenFileName(self, "Open")[0]
        if filepath:
            self.imageOriginal = self.openImageFilename(filepath)
            self.imageScaled = self.scaleImage(self.imageOriginal)
            print("Open image", filepath)
            self.state['imagePath'] = filepath
            self.updateImageAdjustments(reload=True)
            self.updateWindow()
        else:
            print("Open image operation cancelled")

    def saveImage(self):
        """Call this function to save an image file"""
        filters = "JPEG (*.jpg *.jpeg);;PNG (*.png);;All files (*.*)"
        filepath = QFileDialog.getSaveFileName(window, "Save As",
                                               filter=filters)[0]
        if filepath:
            image = self.applyAdjustments(self.imageOriginal)
            image = Image.fromarray((255*image).astype(np.uint8))
            try:
                image.save(filepath)
                print("Save image", filepath)
            except BaseException as e:
                print("Error saving image:", e)
        else:
            print("Save image operation cancelled")

    def openSettings(self):
        """Call this function to open adjustments settings"""
        filters = "JSON (*.json);;All files (*.*)"
        filepath = QFileDialog.getOpenFileName(self, "Open",
                                               filter=filters)[0]
        if filepath:
            try:
                with open(filepath, 'r') as fp:
                    state = json.load(fp)
                assert type(state['imagePath']) is str
                assert type(state['exposure']) in [int, float]
                assert type(state['gamma']) in [int, float]
                assert type(state['hueshift']) in [int, float]
                self.imageOriginal = self.openImageFilename(state['imagePath'])
                self.exposureSlider.set_value(state['exposure'])
                self.gammaSlider.set_value(state['gamma'])
                self.hueshiftSlider.set_value(state['hueshift'])
                self.state = state
                self.imageScaled = self.scaleImage(self.imageOriginal)
                self.updateImageAdjustments(reload=True)
                self.updateWindow()
                print("Settings loaded successfully")
            except BaseException as e:
                print("Error loading settings:", e)
        else:
            print("Open settings operation cancelled")

    def saveSettings(self):
        """Call this function to save adjustments settings"""
        filters = "JSON (*.json);;All files (*.*)"
        filepath = QFileDialog.getSaveFileName(window, "Save As",
                                               filter=filters)[0]
        if filepath:
            try:
                with open(filepath, 'w') as fp:
                    json.dump(self.state, fp, indent=4)
                print("Save settings", filepath)
            except BaseException as e:
                print("Error saving settings:", e)
        else:
            print("Save settings operation cancelled")

    @staticmethod
    def openImageFilename(filename):
        """Load an image as a numpy array"""
        try:
            image = Image.open(filename).convert("RGB")
            data = np.array(image)
            return (data.astype(np.float32)+0.5) / 255.0
        except BaseException as err:
            print(err)
            return None

    @staticmethod
    def scaleImage(image: np.array, target_image_size: int=512) -> np.array:
        """Scale an image to target image size"""
        if type(image) == type(None):
            return None
        img = Image.fromarray((255*image).astype(np.uint8))
        sc = target_image_size / max(image.shape[:2])
        width = round(img.width * sc)
        height = round(img.height * sc)
        resample = Image.BICUBIC if sc < 1.0 else Image.NEAREST
        img = img.resize((width, height), resample=resample)
        return (np.array(img).astype(np.float32)+0.5)/255.0

    def updateDisplayImage(self, data: np.array, reload=False):
        """Display an image in the window, reload layouts/widgets if reload"""
        if type(data) == type(None):
            filename = self.state['imagePath']
            if filename == "":
                message = "Press Ctrl+O to open a file."
            else:
                message = f"Error loading \"{filename}\""
            self.imageContainer = QTextEdit(message)
            self.imageContainer.setReadOnly(True)
            return
        if type(self.imageContainer) != QLabel:
            reload = True
        image = (data*255).astype(np.uint8)
        height, width, dim = image.shape
        assert dim == 3
        qimage = QImage(image, width, height, dim*width, QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        if reload:
            label = QLabel(self.state['imagePath'])
            label.setPixmap(pixmap)
            self.imageContainer = label
        else:
            self.imageContainer.setPixmap(pixmap)

    def updateState(self) -> None:
        """Update adjustments in state from sliders"""
        self.state['exposure'] = self.exposureSlider.get_value()
        self.state['gamma'] = self.gammaSlider.get_value()
        self.state['hueshift'] = self.hueshiftSlider.get_value()

    def applyAdjustments(self, data: np.array) -> np.array:
        """Apply adjustments in state to an image"""
        if type(data) == type(None):
            return None

        # exposure
        exposure = self.state['exposure']
        exposure = 1.0/(1.0-min(exposure,0.999))-1.0
        data = data * exposure

        # gamma
        gamma = self.state['gamma']
        gamma = 1.0/(1.0-min(gamma,0.999))-1.0
        data = pow(data, gamma)

        # hue shift - https://www.shadertoy.com/view/3tjGWm
        hueshift = self.state['hueshift']
        hueshift = np.pi*(2.0*hueshift-1.0)
        cos_hue = np.cos(hueshift)
        sin_hue = 0.57735*np.sin(hueshift)
        hue_m = np.array([cos_hue, -sin_hue, sin_hue])
        hue_m += (1.0-hue_m[0])/3.
        hue_m = np.float32([
            [hue_m[0], hue_m[2], hue_m[1]],
            [hue_m[1], hue_m[0], hue_m[2]],
            [hue_m[2], hue_m[1], hue_m[0]]])
        data = np.matmul(data, hue_m)

        # clamp
        data = np.maximum(np.minimum(data,1.0),0.0)
        return data

    def updateImageAdjustments(self, reload=False):
        """Call this when a slider/parameter changes"""
        self.updateState()
        self.imageAdjusted = self.applyAdjustments(self.imageScaled)
        self.updateDisplayImage(self.imageAdjusted, reload)

    def closeEvent(self, event):
        """Can be used to warn unsaved changes when close window"""
        print("Window closed")
        super().closeEvent(event)


if __name__ == '__main__':

    # https://stackoverflow.com/a/42622179
    def catch_exceptions(t, val, tb):
        QtWidgets.QMessageBox.critical(None,
                                       "An exception was raised",
                                       "Exception type: {}".format(t))
        old_hook(t, val, tb)

    old_hook = sys.excepthook
    sys.excepthook = catch_exceptions

    # run program
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 
