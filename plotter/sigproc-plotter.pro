######################################################################
# Automatically generated by qmake (2.01a) Sat Jul 31 13:42:22 2010
######################################################################

TEMPLATE = app
TARGET = 
DEPENDPATH += .
INCLUDEPATH += . /usr/include/qwt-qt4
LIBS += -lqwt-qt4 -lQtSvg

CONFIG += qt debug

# Input
HEADERS += file_handler.h mainWindow.h \
    openDialogWindow.h
FORMS += plotWidget.ui \
    openDialog.ui
SOURCES += file_handler.cpp mainWindow.cpp plotter.cpp \
    openDialogWindow.cpp
RESOURCES += plotterResources.qrc
