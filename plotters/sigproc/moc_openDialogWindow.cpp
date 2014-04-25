/****************************************************************************
** Meta object code from reading C++ file 'openDialogWindow.h'
**
** Created: Tue Apr 8 12:29:43 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "openDialogWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'openDialogWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_OpenDialogWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      18,   17,   17,   17, 0x08,
      29,   17,   17,   17, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_OpenDialogWindow[] = {
    "OpenDialogWindow\0\0openFile()\0setParams()\0"
};

void OpenDialogWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        OpenDialogWindow *_t = static_cast<OpenDialogWindow *>(_o);
        switch (_id) {
        case 0: _t->openFile(); break;
        case 1: _t->setParams(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData OpenDialogWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject OpenDialogWindow::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_OpenDialogWindow,
      qt_meta_data_OpenDialogWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &OpenDialogWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *OpenDialogWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *OpenDialogWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_OpenDialogWindow))
        return static_cast<void*>(const_cast< OpenDialogWindow*>(this));
    return QDialog::qt_metacast(_clname);
}

int OpenDialogWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
