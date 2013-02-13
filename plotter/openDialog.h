#ifndef OPENDIALOG_H
#define OPENDIALOG_H

#include <ui_openDialog.h>
#include <QDialog>
#include <QString>

class OpenDialog : public QDialog
{
    Q_OBJECT

    public:
        explicit OpenDialog(QWidget *parent = 0);

private slots:
        void openFile();
        void setParams();

private:
        Ui::OpenDialog *_dialog;

    signals:

public:
        QString         filename;
        unsigned        nAntennas, nBits, nChannels, nSamples;
        bool            hasTotalPower;

};

#endif // OPENDIALOG_H
