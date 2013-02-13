#include <QFileDialog>
#include "openDialog.h"
#include "iostream"

OpenDialog::OpenDialog(QWidget *parent) :
    QDialog(parent)
{
    _dialog = new Ui::OpenDialog();
    _dialog->setupUi(this);

    // Create signal connections
    connect(_dialog->pushButton, SIGNAL(clicked()), this, SLOT(openFile()));
    connect(_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(setParams()));
}

void OpenDialog::setParams()
{
    nAntennas = _dialog->antennasEdit->text().toUInt();
    nBits = _dialog->bitsEdit->text().toUInt();
    nChannels = _dialog->channelsEdit->text().toUInt();
    hasTotalPower = _dialog->checkBox->isChecked();
    nSamples = _dialog->sampleEdit->text().toUInt();
}

void OpenDialog::openFile()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (dialog.exec()) {
        filename = dialog.selectedFiles()[0];
        _dialog->filenameEdit->setText(filename);
    }
}
