#ifndef OPENDIALOG_H
#define OPENDIALOG_H

#include <ui_openDialog.h>
#include <QDialog>
#include <QString>

class OpenDialogWindow : public QDialog
{
    Q_OBJECT

    public:
        explicit OpenDialogWindow(QWidget *parent = 0);

private slots:
        void openFile();
        void setParams();

private:
        Ui::OpenDialog *_dialog;

    signals:

public:
        QStringList     filenames;
        float           topFrequency, bandwidth, samplingTime;
        unsigned        nBits, nChannels, nSamples, nBeams;
        bool            hasTotalPower, muLawEncoded;

};

#endif // OPENDIALOG_H
