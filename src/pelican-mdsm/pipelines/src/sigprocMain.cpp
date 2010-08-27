#include "pelican/core/PipelineApplication.h"
#include "LofarTypes.h"
#include "SigprocMdsmPipeline.h"
#include "SigprocAdapter.h"
#include <QtCore/QCoreApplication>
#include <iostream>
#include <map>

using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);

    // Create a PipelineApplication.
    try {
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        pApp.registerPipeline(new SigprocMdsmPipeline);

        // Set the data client.
        pApp.setDataClient("FileDataClient");

        // Start the pipeline driver.
        pApp.start();

    } catch (const QString& error) {
        std::cout << "Error: " << error.toStdString() << std::endl;
    }

    return 0;
}
