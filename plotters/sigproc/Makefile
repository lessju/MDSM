#############################################################################
# Makefile for building: sigproc-plotter
# Generated by qmake (2.01a) (Qt 4.8.1) on: Tue Apr 8 12:29:39 2014
# Project:  sigproc-plotter.pro
# Template: app
# Command: /usr/bin/qmake -unix -o Makefile sigproc-plotter.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_WEBKIT -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED
CFLAGS        = -m64 -pipe -g -D_REENTRANT -Wall -W $(DEFINES)
CXXFLAGS      = -m64 -pipe -fopenmp -g -D_REENTRANT -Wall -W $(DEFINES)
INCPATH       = -I/usr/share/qt4/mkspecs/linux-g++-64 -I. -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4 -I. -I/usr/include/qwt-qt4 -I. -I.
LINK          = g++
LFLAGS        = -m64 -fopenmp
LIBS          = $(SUBLIBS)  -L/usr/lib/x86_64-linux-gnu -lqwt-qt4 -lQtSvg -lgsl -lgslcblas -lQtGui -lQtCore -lpthread 
AR            = ar cqs
RANLIB        = 
QMAKE         = /usr/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = file_handler.cpp \
		mainWindow.cpp \
		plotter.cpp \
		openDialogWindow.cpp moc_mainWindow.cpp \
		moc_openDialogWindow.cpp \
		qrc_plotterResources.cpp
OBJECTS       = file_handler.o \
		mainWindow.o \
		plotter.o \
		openDialogWindow.o \
		moc_mainWindow.o \
		moc_openDialogWindow.o \
		qrc_plotterResources.o
DIST          = /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/debug.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/qt.prf \
		/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/usr/share/qt4/mkspecs/features/moc.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		sigproc-plotter.pro
QMAKE_TARGET  = sigproc-plotter
DESTDIR       = 
TARGET        = sigproc-plotter

first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET): ui_plotWidget.h ui_openDialog.h $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)
	{ test -n "$(DESTDIR)" && DESTDIR="$(DESTDIR)" || DESTDIR=.; } && test $$(gdb --version | sed -e 's,[^0-9]\+\([0-9]\)\.\([0-9]\).*,\1\2,;q') -gt 72 && gdb --nx --batch --quiet -ex 'set confirm off' -ex "save gdb-index $$DESTDIR" -ex quit '$(TARGET)' && test -f $(TARGET).gdb-index && objcopy --add-section '.gdb_index=$(TARGET).gdb-index' --set-section-flags '.gdb_index=readonly' '$(TARGET)' '$(TARGET)' && rm -f $(TARGET).gdb-index || true

Makefile: sigproc-plotter.pro  /usr/share/qt4/mkspecs/linux-g++-64/qmake.conf /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/debug.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/qt.prf \
		/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/usr/share/qt4/mkspecs/features/moc.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		/usr/lib/x86_64-linux-gnu/libQtSvg.prl \
		/usr/lib/x86_64-linux-gnu/libQtGui.prl \
		/usr/lib/x86_64-linux-gnu/libQtCore.prl
	$(QMAKE) -unix -o Makefile sigproc-plotter.pro
/usr/share/qt4/mkspecs/common/unix.conf:
/usr/share/qt4/mkspecs/common/linux.conf:
/usr/share/qt4/mkspecs/common/gcc-base.conf:
/usr/share/qt4/mkspecs/common/gcc-base-unix.conf:
/usr/share/qt4/mkspecs/common/g++-base.conf:
/usr/share/qt4/mkspecs/common/g++-unix.conf:
/usr/share/qt4/mkspecs/qconfig.pri:
/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri:
/usr/share/qt4/mkspecs/features/qt_functions.prf:
/usr/share/qt4/mkspecs/features/qt_config.prf:
/usr/share/qt4/mkspecs/features/exclusive_builds.prf:
/usr/share/qt4/mkspecs/features/default_pre.prf:
/usr/share/qt4/mkspecs/features/debug.prf:
/usr/share/qt4/mkspecs/features/default_post.prf:
/usr/share/qt4/mkspecs/features/qt.prf:
/usr/share/qt4/mkspecs/features/unix/thread.prf:
/usr/share/qt4/mkspecs/features/moc.prf:
/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf:
/usr/share/qt4/mkspecs/features/warn_on.prf:
/usr/share/qt4/mkspecs/features/resources.prf:
/usr/share/qt4/mkspecs/features/uic.prf:
/usr/share/qt4/mkspecs/features/yacc.prf:
/usr/share/qt4/mkspecs/features/lex.prf:
/usr/share/qt4/mkspecs/features/include_source_dir.prf:
/usr/lib/x86_64-linux-gnu/libQtSvg.prl:
/usr/lib/x86_64-linux-gnu/libQtGui.prl:
/usr/lib/x86_64-linux-gnu/libQtCore.prl:
qmake:  FORCE
	@$(QMAKE) -unix -o Makefile sigproc-plotter.pro

dist: 
	@$(CHK_DIR_EXISTS) .tmp/sigproc-plotter1.0.0 || $(MKDIR) .tmp/sigproc-plotter1.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/sigproc-plotter1.0.0/ && $(COPY_FILE) --parents file_handler.h mainWindow.h openDialogWindow.h .tmp/sigproc-plotter1.0.0/ && $(COPY_FILE) --parents plotterResources.qrc .tmp/sigproc-plotter1.0.0/ && $(COPY_FILE) --parents file_handler.cpp mainWindow.cpp plotter.cpp openDialogWindow.cpp .tmp/sigproc-plotter1.0.0/ && $(COPY_FILE) --parents plotWidget.ui openDialog.ui .tmp/sigproc-plotter1.0.0/ && (cd `dirname .tmp/sigproc-plotter1.0.0` && $(TAR) sigproc-plotter1.0.0.tar sigproc-plotter1.0.0 && $(COMPRESS) sigproc-plotter1.0.0.tar) && $(MOVE) `dirname .tmp/sigproc-plotter1.0.0`/sigproc-plotter1.0.0.tar.gz . && $(DEL_FILE) -r .tmp/sigproc-plotter1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


check: first

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

compiler_moc_header_make_all: moc_mainWindow.cpp moc_openDialogWindow.cpp
compiler_moc_header_clean:
	-$(DEL_FILE) moc_mainWindow.cpp moc_openDialogWindow.cpp
moc_mainWindow.cpp: ui_plotWidget.h \
		file_handler.h \
		mainWindow.h
	/usr/bin/moc-qt4 $(DEFINES) $(INCPATH) mainWindow.h -o moc_mainWindow.cpp

moc_openDialogWindow.cpp: ui_openDialog.h \
		openDialogWindow.h
	/usr/bin/moc-qt4 $(DEFINES) $(INCPATH) openDialogWindow.h -o moc_openDialogWindow.cpp

compiler_rcc_make_all: qrc_plotterResources.cpp
compiler_rcc_clean:
	-$(DEL_FILE) qrc_plotterResources.cpp
qrc_plotterResources.cpp: plotterResources.qrc \
		medicina.jpg
	/usr/bin/rcc -name plotterResources plotterResources.qrc -o qrc_plotterResources.cpp

compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_uic_make_all: ui_plotWidget.h ui_openDialog.h
compiler_uic_clean:
	-$(DEL_FILE) ui_plotWidget.h ui_openDialog.h
ui_plotWidget.h: plotWidget.ui
	/usr/bin/uic-qt4 plotWidget.ui -o ui_plotWidget.h

ui_openDialog.h: openDialog.ui
	/usr/bin/uic-qt4 openDialog.ui -o ui_openDialog.h

compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: compiler_moc_header_clean compiler_rcc_clean compiler_uic_clean 

####### Compile

file_handler.o: file_handler.cpp file_handler.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o file_handler.o file_handler.cpp

mainWindow.o: mainWindow.cpp ui_openDialog.h \
		openDialogWindow.h \
		mainWindow.h \
		ui_plotWidget.h \
		file_handler.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o mainWindow.o mainWindow.cpp

plotter.o: plotter.cpp mainWindow.h \
		ui_plotWidget.h \
		file_handler.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o plotter.o plotter.cpp

openDialogWindow.o: openDialogWindow.cpp openDialogWindow.h \
		ui_openDialog.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o openDialogWindow.o openDialogWindow.cpp

moc_mainWindow.o: moc_mainWindow.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_mainWindow.o moc_mainWindow.cpp

moc_openDialogWindow.o: moc_openDialogWindow.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_openDialogWindow.o moc_openDialogWindow.cpp

qrc_plotterResources.o: qrc_plotterResources.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o qrc_plotterResources.o qrc_plotterResources.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

