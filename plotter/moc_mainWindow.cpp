/****************************************************************************
** Meta object code from reading C++ file 'mainWindow.h'
**
** Created: Tue Jan 8 18:37:17 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.6.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.6.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       4,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x08,
      23,   11,   11,   11, 0x08,
      35,   11,   11,   11, 0x08,
      48,   11,   11,   11, 0x08,
      61,   55,   11,   11, 0x08,
      72,   11,   11,   11, 0x28,
      81,   79,   11,   11, 0x08,
      98,   79,   11,   11, 0x08,
     121,   79,   11,   11, 0x08,
     138,   79,   11,   11, 0x08,
     154,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0\0openFile()\0liveFiles()\0"
    "saveBuffer()\0quit()\0reset\0plot(bool)\0"
    "plot()\0i\0sliderMoved(int)\0"
    "beamNumberChanged(int)\0plotChannel(int)\0"
    "sampleSpin(int)\0initialisePlotter()\0"
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: openFile(); break;
        case 1: liveFiles(); break;
        case 2: saveBuffer(); break;
        case 3: quit(); break;
        case 4: plot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: plot(); break;
        case 6: sliderMoved((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: beamNumberChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: plotChannel((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: sampleSpin((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: initialisePlotter(); break;
        default: ;
        }
        _id -= 11;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
