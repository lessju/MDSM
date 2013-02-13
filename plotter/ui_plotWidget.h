/********************************************************************************
** Form generated from reading UI file 'plotWidget.ui'
**
** Created: Tue Nov 13 11:35:21 2012
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
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
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
    QSpacerItem *verticalSpacer;
    QPushButton *plotButton;

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
        tabWidget->addTab(tab, QString());

        gridLayout->addWidget(tabWidget, 1, 1, 4, 1);

        verticalGroupBox = new QVBoxLayout();
        verticalGroupBox->setObjectName(QString::fromUtf8("verticalGroupBox"));
        verticalGroupBox->setSizeConstraint(QLayout::SetMinimumSize);
        groupBox = new QGroupBox(gridLayoutWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setEnabled(false);
        gridLayoutWidget_2 = new QWidget(groupBox);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(0, 20, 341, 71));
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
        integrationBox->setMaximum(4096);

        gridLayout_2->addWidget(integrationBox, 0, 1, 1, 2);

        beamNumber = new QSpinBox(gridLayoutWidget_2);
        beamNumber->setObjectName(QString::fromUtf8("beamNumber"));
        beamNumber->setMinimum(1);
        beamNumber->setMaximum(8);

        gridLayout_2->addWidget(beamNumber, 1, 1, 1, 2);

        gridLayout_2->setColumnStretch(0, 1);
        gridLayout_2->setColumnStretch(1, 1);

        verticalGroupBox->addWidget(groupBox);

        groupBox_2 = new QGroupBox(gridLayoutWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setEnabled(false);
        gridLayoutWidget_3 = new QWidget(groupBox_2);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(0, 20, 341, 91));
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

        gridLayout_3->addWidget(timeSlider, 0, 0, 1, 2);

        sampleSpin = new QSpinBox(gridLayoutWidget_3);
        sampleSpin->setObjectName(QString::fromUtf8("sampleSpin"));
        sampleSpin->setSingleStep(100);
        sampleSpin->setValue(0);

        gridLayout_3->addWidget(sampleSpin, 1, 1, 1, 1);

        current_time_label = new QLabel(gridLayoutWidget_3);
        current_time_label->setObjectName(QString::fromUtf8("current_time_label"));
        current_time_label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        current_time_label->setIndent(20);

        gridLayout_3->addWidget(current_time_label, 2, 0, 1, 2);

        gridLayout_3->setColumnStretch(0, 1);
        gridLayout_3->setColumnStretch(1, 2);

        verticalGroupBox->addWidget(groupBox_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalGroupBox->addItem(verticalSpacer);

        plotButton = new QPushButton(gridLayoutWidget);
        plotButton->setObjectName(QString::fromUtf8("plotButton"));
        plotButton->setEnabled(false);

        verticalGroupBox->addWidget(plotButton);

        verticalGroupBox->setStretch(0, 1);
        verticalGroupBox->setStretch(1, 1);
        verticalGroupBox->setStretch(2, 3);
        verticalGroupBox->setStretch(3, 1);

        gridLayout->addLayout(verticalGroupBox, 1, 0, 4, 1);

        gridLayout->setColumnStretch(0, 4);
        gridLayout->setColumnStretch(1, 8);
        QWidget::setTabOrder(tabWidget, channelSpin);

        retranslateUi(SigprocPlotter);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(SigprocPlotter);
    } // setupUi

    void retranslateUi(QWidget *SigprocPlotter)
    {
        SigprocPlotter->setWindowTitle(QApplication::translate("SigprocPlotter", "SigprocPlotter", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(specTab), QApplication::translate("SigprocPlotter", "Spectogram", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("SigprocPlotter", "Channel Number", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(chanTab), QApplication::translate("SigprocPlotter", "Channel Intensity", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("SigprocPlotter", "Bandpass", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("SigprocPlotter", "Plotter Settings", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("SigprocPlotter", "Integrate:", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("SigprocPlotter", "Beam Number:", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("SigprocPlotter", "Time Control", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("SigprocPlotter", "Finer Control", 0, QApplication::UnicodeUTF8));
        current_time_label->setText(QApplication::translate("SigprocPlotter", "0 s", 0, QApplication::UnicodeUTF8));
        plotButton->setText(QApplication::translate("SigprocPlotter", "Plot", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SigprocPlotter: public Ui_SigprocPlotter {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PLOTWIDGET_H
