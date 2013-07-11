#-------------------------------------------------
#
# Project created by QtCreator 2013-07-04T15:48:35
#
#-------------------------------------------------

QT       += core

QT       -= gui

LIBS     += -fopenmp

QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp

TARGET = Pulse_Simulator
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp
