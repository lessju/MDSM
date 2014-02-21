#include "mainWindow.h"
#include <QApplication>
#include <QWidget>
#include "file_handler.h"

int main(int argc, char *argv[]) 
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();

    return app.exec();
}
