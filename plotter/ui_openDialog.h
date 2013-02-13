/********************************************************************************
** Form generated from reading UI file 'openDialog.ui'
**
** Created: Tue Jan 8 18:34:57 2013
**      by: Qt User Interface Compiler version 4.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_OPENDIALOG_H
#define UI_OPENDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OpenDialog
{
public:
    QDialogButtonBox *buttonBox;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QLabel *label_3;
    QLabel *label_4;
    QLineEdit *frequency_edit;
    QLineEdit *bitsEdit;
    QLineEdit *filenameEdit;
    QPushButton *pushButton;
    QCheckBox *checkBox;
    QLabel *label_5;
    QLineEdit *sampleEdit;
    QLabel *label_6;
    QLineEdit *bandwidth_edit;
    QLabel *label_7;
    QLineEdit *channels_edit;
    QLabel *label_8;
    QLineEdit *sampling_edit;
    QLabel *label_9;
    QLineEdit *beamsEdit;
    QCheckBox *muLawEncoded_checkbox;

    void setupUi(QDialog *OpenDialog)
    {
        if (OpenDialog->objectName().isEmpty())
            OpenDialog->setObjectName(QString::fromUtf8("OpenDialog"));
        OpenDialog->resize(319, 399);
        buttonBox = new QDialogButtonBox(OpenDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(-40, 360, 351, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        gridLayoutWidget = new QWidget(OpenDialog);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(10, 10, 298, 341));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setHorizontalSpacing(10);
        gridLayout->setContentsMargins(10, 0, 5, 0);
        label_3 = new QLabel(gridLayoutWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 1, 0, 1, 1);

        label_4 = new QLabel(gridLayoutWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 6, 0, 1, 1);

        frequency_edit = new QLineEdit(gridLayoutWidget);
        frequency_edit->setObjectName(QString::fromUtf8("frequency_edit"));

        gridLayout->addWidget(frequency_edit, 1, 1, 1, 2);

        bitsEdit = new QLineEdit(gridLayoutWidget);
        bitsEdit->setObjectName(QString::fromUtf8("bitsEdit"));

        gridLayout->addWidget(bitsEdit, 6, 1, 1, 2);

        filenameEdit = new QLineEdit(gridLayoutWidget);
        filenameEdit->setObjectName(QString::fromUtf8("filenameEdit"));

        gridLayout->addWidget(filenameEdit, 0, 0, 1, 2);

        pushButton = new QPushButton(gridLayoutWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setMaximumSize(QSize(25, 16777215));

        gridLayout->addWidget(pushButton, 0, 2, 1, 1);

        checkBox = new QCheckBox(gridLayoutWidget);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));
        checkBox->setChecked(true);

        gridLayout->addWidget(checkBox, 10, 0, 1, 3);

        label_5 = new QLabel(gridLayoutWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 7, 0, 1, 1);

        sampleEdit = new QLineEdit(gridLayoutWidget);
        sampleEdit->setObjectName(QString::fromUtf8("sampleEdit"));

        gridLayout->addWidget(sampleEdit, 7, 1, 1, 2);

        label_6 = new QLabel(gridLayoutWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 2, 0, 1, 1);

        bandwidth_edit = new QLineEdit(gridLayoutWidget);
        bandwidth_edit->setObjectName(QString::fromUtf8("bandwidth_edit"));

        gridLayout->addWidget(bandwidth_edit, 2, 1, 1, 2);

        label_7 = new QLabel(gridLayoutWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout->addWidget(label_7, 3, 0, 1, 1);

        channels_edit = new QLineEdit(gridLayoutWidget);
        channels_edit->setObjectName(QString::fromUtf8("channels_edit"));

        gridLayout->addWidget(channels_edit, 3, 1, 1, 2);

        label_8 = new QLabel(gridLayoutWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout->addWidget(label_8, 4, 0, 1, 1);

        sampling_edit = new QLineEdit(gridLayoutWidget);
        sampling_edit->setObjectName(QString::fromUtf8("sampling_edit"));

        gridLayout->addWidget(sampling_edit, 4, 1, 1, 2);

        label_9 = new QLabel(gridLayoutWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout->addWidget(label_9, 5, 0, 1, 1);

        beamsEdit = new QLineEdit(gridLayoutWidget);
        beamsEdit->setObjectName(QString::fromUtf8("beamsEdit"));

        gridLayout->addWidget(beamsEdit, 5, 1, 1, 2);

        muLawEncoded_checkbox = new QCheckBox(gridLayoutWidget);
        muLawEncoded_checkbox->setObjectName(QString::fromUtf8("muLawEncoded_checkbox"));
        muLawEncoded_checkbox->setChecked(false);

        gridLayout->addWidget(muLawEncoded_checkbox, 9, 0, 1, 3);

        gridLayout->setColumnStretch(0, 2);
        QWidget::setTabOrder(filenameEdit, pushButton);
        QWidget::setTabOrder(pushButton, frequency_edit);
        QWidget::setTabOrder(frequency_edit, bitsEdit);
        QWidget::setTabOrder(bitsEdit, sampleEdit);
        QWidget::setTabOrder(sampleEdit, checkBox);
        QWidget::setTabOrder(checkBox, buttonBox);

        retranslateUi(OpenDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), OpenDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), OpenDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(OpenDialog);
    } // setupUi

    void retranslateUi(QDialog *OpenDialog)
    {
        OpenDialog->setWindowTitle(QApplication::translate("OpenDialog", "Open Dialog", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("OpenDialog", "Top Frequency", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("OpenDialog", "Bits per sample", 0, QApplication::UnicodeUTF8));
        frequency_edit->setText(QApplication::translate("OpenDialog", "413.8984375", 0, QApplication::UnicodeUTF8));
        bitsEdit->setText(QApplication::translate("OpenDialog", "32", 0, QApplication::UnicodeUTF8));
        filenameEdit->setText(QApplication::translate("OpenDialog", "Filepaths...", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("OpenDialog", "...", 0, QApplication::UnicodeUTF8));
        checkBox->setText(QApplication::translate("OpenDialog", "Values have Total Power", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("OpenDialog", "Plot samples", 0, QApplication::UnicodeUTF8));
        sampleEdit->setText(QApplication::translate("OpenDialog", "500", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("OpenDialog", "Bandwidth", 0, QApplication::UnicodeUTF8));
        bandwidth_edit->setText(QApplication::translate("OpenDialog", "10", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("OpenDialog", "Number of Channels", 0, QApplication::UnicodeUTF8));
        channels_edit->setText(QApplication::translate("OpenDialog", "512", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("OpenDialog", "Sampling Time", 0, QApplication::UnicodeUTF8));
        sampling_edit->setText(QApplication::translate("OpenDialog", "0.0001024", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("OpenDialog", "Number of Beams", 0, QApplication::UnicodeUTF8));
        beamsEdit->setText(QApplication::translate("OpenDialog", "1", 0, QApplication::UnicodeUTF8));
        muLawEncoded_checkbox->setText(QApplication::translate("OpenDialog", "Values are Mu-Law Encoded", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class OpenDialog: public Ui_OpenDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_OPENDIALOG_H
