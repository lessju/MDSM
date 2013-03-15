/********************************************************************************
** Form generated from reading UI file 'plotWidget.ui'
**
** Created: Wed Mar 13 16:45:52 2013
**      by: Qt User Interface Compiler version 4.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PLOTWIDGET_H
#define UI_PLOTWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "qwt_plot.h"

QT_BEGIN_NAMESPACE

class Ui_SigprocPlotter
{
public:
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QVBoxLayout *verticalGroupBox;
    QGroupBox *groupBox;
    QWidget *gridLayoutWidget_2;
    QGridLayout *gridLayout_2;
    QLabel *label_3;
    QLabel *label_7;
    QSpinBox *integrationBox;
    QSpinBox *beamNumber;
    QGroupBox *groupBox_2;
    QWidget *gridLayoutWidget_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QSlider *timeSlider;
    QSpinBox *sampleSpin;
    QLabel *current_time_label;
    QGroupBox *groupBox_3;
    QWidget *gridLayoutWidget_4;
    QGridLayout *gridLayout_4;
    QLabel *label_6;
    QLabel *label_8;
    QDoubleSpinBox *dmSpinBox;
    QDoubleSpinBox *periodSpinBox;
    QSpinBox *foldingSpinBox;
    QLabel *label_9;
    QPushButton *plotButton;
    QGroupBox *groupBox_4;
    QWidget *gridLayoutWidget_5;
    QGridLayout *gridLayout_5;
    QDoubleSpinBox *channelThresholdBox;
    QCheckBox *channelRfiBox;
    QCheckBox *spectrumRfiBox;
    QLabel *label;
    QLabel *label_10;
    QSpinBox *channelBlockBox;
    QLabel *label_2;
    QDoubleSpinBox *spectrumThresholdBox;
    QLabel *label_11;
    QSpinBox *fitDegreesBox;
    QTabWidget *tabWidget;
    QWidget *specTab;
    QwtPlot *specPlot;
    QWidget *chanTab;
    QwtPlot *chanPlot;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QLabel *label_4;
    QSpinBox *channelSpin;
    QWidget *tab;
    QwtPlot *bandPlot;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_12;
    QLineEdit *channelMaskEdit;
    QWidget *timeTab;
    QwtPlot *timePlot;

    void setupUi(QWidget *SigprocPlotter)
    {
        if (SigprocPlotter->objectName().isEmpty())
            SigprocPlotter->setObjectName(QString::fromUtf8("SigprocPlotter"));
        SigprocPlotter->setEnabled(true);
        SigprocPlotter->resize(1066, 614);
        gridLayoutWidget = new QWidget(SigprocPlotter);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(0, 0, 1061, 612));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setSizeConstraint(QLayout::SetNoConstraint);
        gridLayout->setHorizontalSpacing(10);
        gridLayout->setVerticalSpacing(0);
        gridLayout->setContentsMargins(6, 30, 6, 6);
        verticalGroupBox = new QVBoxLayout();
        verticalGroupBox->setObjectName(QString::fromUtf8("verticalGroupBox"));
        verticalGroupBox->setSizeConstraint(QLayout::SetMinimumSize);
        groupBox = new QGroupBox(gridLayoutWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setEnabled(false);
        gridLayoutWidget_2 = new QWidget(groupBox);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(0, 20, 341, 68));
        gridLayout_2 = new QGridLayout(gridLayoutWidget_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setHorizontalSpacing(10);
        gridLayout_2->setContentsMargins(10, 0, 5, 0);
        label_3 = new QLabel(gridLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 0, 0, 1, 1);

        label_7 = new QLabel(gridLayoutWidget_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_2->addWidget(label_7, 1, 0, 2, 1);

        integrationBox = new QSpinBox(gridLayoutWidget_2);
        integrationBox->setObjectName(QString::fromUtf8("integrationBox"));
        integrationBox->setMinimum(1);
        integrationBox->setMaximum(4096);

        gridLayout_2->addWidget(integrationBox, 0, 1, 1, 2);

        beamNumber = new QSpinBox(gridLayoutWidget_2);
        beamNumber->setObjectName(QString::fromUtf8("beamNumber"));
        beamNumber->setMinimum(1);
        beamNumber->setMaximum(1);

        gridLayout_2->addWidget(beamNumber, 1, 1, 1, 2);

        gridLayout_2->setColumnStretch(0, 1);
        gridLayout_2->setColumnStretch(1, 1);

        verticalGroupBox->addWidget(groupBox);

        groupBox_2 = new QGroupBox(gridLayoutWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setEnabled(false);
        gridLayoutWidget_3 = new QWidget(groupBox_2);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(0, 20, 341, 71));
        gridLayout_3 = new QGridLayout(gridLayoutWidget_3);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setHorizontalSpacing(10);
        gridLayout_3->setContentsMargins(10, 0, 5, 0);
        label_5 = new QLabel(gridLayoutWidget_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        timeSlider = new QSlider(gridLayoutWidget_3);
        timeSlider->setObjectName(QString::fromUtf8("timeSlider"));
        timeSlider->setMaximum(1000);
        timeSlider->setOrientation(Qt::Horizontal);
        timeSlider->setTickPosition(QSlider::TicksBelow);
        timeSlider->setTickInterval(100);

        gridLayout_3->addWidget(timeSlider, 0, 0, 1, 3);

        sampleSpin = new QSpinBox(gridLayoutWidget_3);
        sampleSpin->setObjectName(QString::fromUtf8("sampleSpin"));
        sampleSpin->setSingleStep(100);
        sampleSpin->setValue(0);

        gridLayout_3->addWidget(sampleSpin, 1, 1, 1, 1);

        current_time_label = new QLabel(gridLayoutWidget_3);
        current_time_label->setObjectName(QString::fromUtf8("current_time_label"));
        current_time_label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        current_time_label->setIndent(20);

        gridLayout_3->addWidget(current_time_label, 1, 2, 1, 1);

        gridLayout_3->setColumnStretch(0, 1);
        gridLayout_3->setColumnStretch(1, 4);
        gridLayout_3->setColumnStretch(2, 2);

        verticalGroupBox->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(gridLayoutWidget);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        groupBox_3->setEnabled(false);
        gridLayoutWidget_4 = new QWidget(groupBox_3);
        gridLayoutWidget_4->setObjectName(QString::fromUtf8("gridLayoutWidget_4"));
        gridLayoutWidget_4->setGeometry(QRect(-1, 19, 341, 128));
        gridLayout_4 = new QGridLayout(gridLayoutWidget_4);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setHorizontalSpacing(10);
        gridLayout_4->setContentsMargins(10, 0, 5, 0);
        label_6 = new QLabel(gridLayoutWidget_4);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout_4->addWidget(label_6, 0, 0, 1, 1);

        label_8 = new QLabel(gridLayoutWidget_4);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_4->addWidget(label_8, 1, 0, 1, 1);

        dmSpinBox = new QDoubleSpinBox(gridLayoutWidget_4);
        dmSpinBox->setObjectName(QString::fromUtf8("dmSpinBox"));
        dmSpinBox->setDecimals(3);

        gridLayout_4->addWidget(dmSpinBox, 0, 1, 1, 1);

        periodSpinBox = new QDoubleSpinBox(gridLayoutWidget_4);
        periodSpinBox->setObjectName(QString::fromUtf8("periodSpinBox"));
        periodSpinBox->setDecimals(4);
        periodSpinBox->setMaximum(5000);
        periodSpinBox->setSingleStep(10);

        gridLayout_4->addWidget(periodSpinBox, 1, 1, 1, 1);

        foldingSpinBox = new QSpinBox(gridLayoutWidget_4);
        foldingSpinBox->setObjectName(QString::fromUtf8("foldingSpinBox"));
        foldingSpinBox->setMinimum(1);
        foldingSpinBox->setMaximum(1000);

        gridLayout_4->addWidget(foldingSpinBox, 2, 1, 1, 1);

        label_9 = new QLabel(gridLayoutWidget_4);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_4->addWidget(label_9, 2, 0, 1, 1);

        plotButton = new QPushButton(gridLayoutWidget_4);
        plotButton->setObjectName(QString::fromUtf8("plotButton"));

        gridLayout_4->addWidget(plotButton, 3, 0, 1, 2);


        verticalGroupBox->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(gridLayoutWidget);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        groupBox_4->setEnabled(false);
        gridLayoutWidget_5 = new QWidget(groupBox_4);
        gridLayoutWidget_5->setObjectName(QString::fromUtf8("gridLayoutWidget_5"));
        gridLayoutWidget_5->setGeometry(QRect(0, 20, 341, 181));
        gridLayout_5 = new QGridLayout(gridLayoutWidget_5);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_5->setHorizontalSpacing(10);
        gridLayout_5->setContentsMargins(10, 0, 5, 0);
        channelThresholdBox = new QDoubleSpinBox(gridLayoutWidget_5);
        channelThresholdBox->setObjectName(QString::fromUtf8("channelThresholdBox"));
        channelThresholdBox->setEnabled(false);
        channelThresholdBox->setDecimals(2);
        channelThresholdBox->setMinimum(1);
        channelThresholdBox->setMaximum(100);
        channelThresholdBox->setSingleStep(0.1);

        gridLayout_5->addWidget(channelThresholdBox, 3, 1, 1, 1);

        channelRfiBox = new QCheckBox(gridLayoutWidget_5);
        channelRfiBox->setObjectName(QString::fromUtf8("channelRfiBox"));

        gridLayout_5->addWidget(channelRfiBox, 0, 0, 1, 1);

        spectrumRfiBox = new QCheckBox(gridLayoutWidget_5);
        spectrumRfiBox->setObjectName(QString::fromUtf8("spectrumRfiBox"));

        gridLayout_5->addWidget(spectrumRfiBox, 0, 1, 1, 1);

        label = new QLabel(gridLayoutWidget_5);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_5->addWidget(label, 3, 0, 1, 1);

        label_10 = new QLabel(gridLayoutWidget_5);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_5->addWidget(label_10, 5, 0, 1, 1);

        channelBlockBox = new QSpinBox(gridLayoutWidget_5);
        channelBlockBox->setObjectName(QString::fromUtf8("channelBlockBox"));
        channelBlockBox->setEnabled(false);
        channelBlockBox->setMinimum(1);
        channelBlockBox->setMaximum(2000);
        channelBlockBox->setSingleStep(5);
        channelBlockBox->setValue(1);

        gridLayout_5->addWidget(channelBlockBox, 5, 1, 1, 1);

        label_2 = new QLabel(gridLayoutWidget_5);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_5->addWidget(label_2, 2, 0, 1, 1);

        spectrumThresholdBox = new QDoubleSpinBox(gridLayoutWidget_5);
        spectrumThresholdBox->setObjectName(QString::fromUtf8("spectrumThresholdBox"));
        spectrumThresholdBox->setMinimum(1);
        spectrumThresholdBox->setMaximum(100);
        spectrumThresholdBox->setSingleStep(0.5);

        gridLayout_5->addWidget(spectrumThresholdBox, 2, 1, 1, 1);

        label_11 = new QLabel(gridLayoutWidget_5);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_5->addWidget(label_11, 1, 0, 1, 1);

        fitDegreesBox = new QSpinBox(gridLayoutWidget_5);
        fitDegreesBox->setObjectName(QString::fromUtf8("fitDegreesBox"));
        fitDegreesBox->setEnabled(false);
        fitDegreesBox->setMinimum(1);
        fitDegreesBox->setMaximum(20);
        fitDegreesBox->setSingleStep(1);
        fitDegreesBox->setValue(12);

        gridLayout_5->addWidget(fitDegreesBox, 1, 1, 1, 1);


        verticalGroupBox->addWidget(groupBox_4);

        verticalGroupBox->setStretch(0, 5);
        verticalGroupBox->setStretch(1, 5);
        verticalGroupBox->setStretch(2, 8);
        verticalGroupBox->setStretch(3, 10);

        gridLayout->addLayout(verticalGroupBox, 1, 0, 4, 1);

        tabWidget = new QTabWidget(gridLayoutWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setMovable(true);
        specTab = new QWidget();
        specTab->setObjectName(QString::fromUtf8("specTab"));
        specPlot = new QwtPlot(specTab);
        specPlot->setObjectName(QString::fromUtf8("specPlot"));
        specPlot->setGeometry(QRect(0, 0, 681, 501));
        specPlot->setMinimumSize(QSize(500, 0));
        specPlot->setCursor(QCursor(Qt::CrossCursor));
        tabWidget->addTab(specTab, QString());
        chanTab = new QWidget();
        chanTab->setObjectName(QString::fromUtf8("chanTab"));
        chanPlot = new QwtPlot(chanTab);
        chanPlot->setObjectName(QString::fromUtf8("chanPlot"));
        chanPlot->setGeometry(QRect(0, 0, 681, 461));
        chanPlot->setMinimumSize(QSize(500, 0));
        chanPlot->setCursor(QCursor(Qt::CrossCursor));
        layoutWidget = new QWidget(chanTab);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(20, 470, 262, 31));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        label_4 = new QLabel(layoutWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout->addWidget(label_4);

        channelSpin = new QSpinBox(layoutWidget);
        channelSpin->setObjectName(QString::fromUtf8("channelSpin"));
        channelSpin->setEnabled(false);
        channelSpin->setMinimum(1);

        horizontalLayout->addWidget(channelSpin);

        tabWidget->addTab(chanTab, QString());
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        bandPlot = new QwtPlot(tab);
        bandPlot->setObjectName(QString::fromUtf8("bandPlot"));
        bandPlot->setGeometry(QRect(0, 0, 681, 501));
        bandPlot->setMinimumSize(QSize(500, 0));
        bandPlot->setCursor(QCursor(Qt::CrossCursor));
        horizontalLayoutWidget = new QWidget(tab);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(0, 500, 471, 41));
        horizontalLayout_2 = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        label_12 = new QLabel(horizontalLayoutWidget);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setEnabled(false);

        horizontalLayout_2->addWidget(label_12);

        channelMaskEdit = new QLineEdit(horizontalLayoutWidget);
        channelMaskEdit->setObjectName(QString::fromUtf8("channelMaskEdit"));
        channelMaskEdit->setEnabled(false);

        horizontalLayout_2->addWidget(channelMaskEdit);

        tabWidget->addTab(tab, QString());
        timeTab = new QWidget();
        timeTab->setObjectName(QString::fromUtf8("timeTab"));
        timePlot = new QwtPlot(timeTab);
        timePlot->setObjectName(QString::fromUtf8("timePlot"));
        timePlot->setGeometry(QRect(0, 0, 681, 501));
        timePlot->setMinimumSize(QSize(500, 0));
        timePlot->setCursor(QCursor(Qt::CrossCursor));
        tabWidget->addTab(timeTab, QString());

        gridLayout->addWidget(tabWidget, 1, 1, 4, 1);

        gridLayout->setColumnStretch(0, 4);
        gridLayout->setColumnStretch(1, 8);
        QWidget::setTabOrder(tabWidget, channelSpin);

        retranslateUi(SigprocPlotter);
        QObject::connect(channelRfiBox, SIGNAL(toggled(bool)), fitDegreesBox, SLOT(setEnabled(bool)));
        QObject::connect(spectrumRfiBox, SIGNAL(toggled(bool)), fitDegreesBox, SLOT(setEnabled(bool)));
        QObject::connect(channelRfiBox, SIGNAL(toggled(bool)), channelThresholdBox, SLOT(setEnabled(bool)));
        QObject::connect(channelRfiBox, SIGNAL(toggled(bool)), channelBlockBox, SLOT(setEnabled(bool)));
        QObject::connect(spectrumRfiBox, SIGNAL(toggled(bool)), spectrumThresholdBox, SLOT(setEnabled(bool)));
        QObject::connect(channelRfiBox, SIGNAL(toggled(bool)), label_12, SLOT(setEnabled(bool)));
        QObject::connect(channelRfiBox, SIGNAL(toggled(bool)), channelMaskEdit, SLOT(setEnabled(bool)));

        tabWidget->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(SigprocPlotter);
    } // setupUi

    void retranslateUi(QWidget *SigprocPlotter)
    {
        SigprocPlotter->setWindowTitle(QApplication::translate("SigprocPlotter", "SigprocPlotter", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("SigprocPlotter", "Plotter Settings", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("SigprocPlotter", "Integrate:", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("SigprocPlotter", "Beam Number:", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("SigprocPlotter", "Time Control", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("SigprocPlotter", "Finer Control:", 0, QApplication::UnicodeUTF8));
        current_time_label->setText(QApplication::translate("SigprocPlotter", "0 s", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("SigprocPlotter", "Pulse utilities", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("SigprocPlotter", "Dedisperse (DM)", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("SigprocPlotter", "Period (ms)", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("SigprocPlotter", "Number of profiles", 0, QApplication::UnicodeUTF8));
        plotButton->setText(QApplication::translate("SigprocPlotter", "Toggle Plot Type (Data)", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("SigprocPlotter", "RFI utilities", 0, QApplication::UnicodeUTF8));
        channelRfiBox->setText(QApplication::translate("SigprocPlotter", "Channel Clipper", 0, QApplication::UnicodeUTF8));
        spectrumRfiBox->setText(QApplication::translate("SigprocPlotter", "Spectrum Clipper", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("SigprocPlotter", "Channel Threshold", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("SigprocPlotter", "Channel block length", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("SigprocPlotter", "Spectrum Threshold", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("SigprocPlotter", "Polynomial Fit Degrees", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(specTab), QApplication::translate("SigprocPlotter", "Spectogram", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("SigprocPlotter", "Channel Number", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(chanTab), QApplication::translate("SigprocPlotter", "Channel Intensity", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("SigprocPlotter", "Clip Specific Channels (comma-separated:", 0, QApplication::UnicodeUTF8));
        channelMaskEdit->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("SigprocPlotter", "Bandpass", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(timeTab), QApplication::translate("SigprocPlotter", "Time Series", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SigprocPlotter: public Ui_SigprocPlotter {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PLOTWIDGET_H
