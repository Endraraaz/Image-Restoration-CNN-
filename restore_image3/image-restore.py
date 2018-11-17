# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!
import os

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import easygui
import Tkinter,tkFileDialog

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(711, 743)
        font = QtGui.QFont()
        font.setPointSize(12)
        Form.setFont(font)
        self.pushButton = QtGui.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 161, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.label = QtGui.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(200, 520, 271, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_1 = QtGui.QLabel(Form)
        self.label_1.setGeometry(QtCore.QRect(40, 50, 631, 461))
        self.label_1.setObjectName(_fromUtf8("label_1"))

        self.pushButton_2 = QtGui.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 560, 161, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 690, 161, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        
        self.pushButton_4 = QtGui.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(425, 690, 161, 27))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_3"))

        self.textBrowser = QtGui.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(40, 620, 611, 31))
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):

        Form.setWindowTitle(_translate("Form", "Image Restoration Software", None))
        self.pushButton.setText(_translate("Form", "Input Image Here", None))
        self.label.setText(_translate("Form", "                       Original Image File", None))
        #self.label_1.setText(_translate("Form","image", None))
        self.pushButton_2.setText(_translate("Form", "Output Directory", None))
        self.pushButton_3.setText(_translate("Form", "Super-resolution!", None))
        self.pushButton_4.setText(_translate("Form", "Fill-the-void!", None))
        self.pushButton.clicked.connect(self.OpenClick)
        self.pushButton_2.clicked.connect(self.OpenDir)
        self.pushButton_3.clicked.connect(self.open_restore)
        self.pushButton_4.clicked.connect(self.open_inpaint)
    
    def open_restore(self):
        os.system('python super-resolution.py'+" "+imagepath+" "+dirname)

    def open_inpaint(self):
        os.system('python3 put_mask.py'+" "+imagepath+" "+dirname)

   
    def OpenClick(self):
         global imagepath
         imagepath = easygui.fileopenbox(msg='Please select image file',title='Specify File',filetypes=["*.jpg","*.png"])
         #imagepath=path[0]
         pixmap=QPixmap(imagepath)
         self.label_1.setPixmap(pixmap)
         # self.label_1.resize()
         
    def OpenDir(self):
        global dirname
        root = Tkinter.Tk()
        root.withdraw()
        dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
        if dirname:
           self.textBrowser.clear()
           self.textBrowser.append(str(dirname))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Form = QtGui.QWidget()
    p = Form.palette()
    p.setColor(Form.backgroundRole(), QColor(126,150,150))
    Form.setPalette(p)
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

