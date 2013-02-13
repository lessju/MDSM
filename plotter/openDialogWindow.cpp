#include <QFileDialog>
#include "openDialogWindow.h"
#include "iostream"

OpenDialogWindow::OpenDialogWindow(QWidget *parent) :
    QDialog(parent)
{
    _dialog = new Ui::OpenDialog();
    _dialog->setupUi(this);

    // Create signal connections
    connect(_dialog->pushButton, SIGNAL(clicked()), this, SLOT(openFile()));
    connect(_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(setParams()));
}

void OpenDialogWindow::setParams()
{
    topFrequency = _dialog -> frequency_edit -> text().toFloat();
    bandwidth = _dialog -> bandwidth_edit -> text().toFloat();
    samplingTime = _dialog -> sampling_edit -> text().toFloat();
    nBits = _dialog -> bitsEdit -> text().toUInt();
    nChannels = _dialog -> channels_edit -> text().toUInt();
    hasTotalPower = _dialog -> checkBox -> isChecked();
    nSamples = _dialog -> sampleEdit -> text().toUInt();
    nBeams = _dialog -> beamsEdit -> text().toUInt();
    muLawEncoded = _dialog -> muLawEncoded_checkbox -> isChecked();
}

void OpenDialogWindow::openFile()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::ExistingFiles);

    if (dialog.exec()) {
        filenames = dialog.selectedFiles();
        if (filenames.length() == 0)
            return;
        _dialog->filenameEdit->setText(filenames[0]);
    }
}
