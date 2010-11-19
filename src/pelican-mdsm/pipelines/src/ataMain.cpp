#include "pelican/core/PipelineApplication.h"
#include "LofarTypes.h"
#include "AtaMdsmPipeline.h"
#include "RawVoltageAdapter.h"
#include "DedispersedDataWriter.h"
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
        pApp.registerPipeline(new AtaMdsmPipeline);

        // Set the data client.
        pApp.setDataClient("FileDataClient");

        // Start the pipeline driver.
        pApp.start();

    } catch (const QString& error) {
        std::cout << "Error: " << error.toStdString() << std::endl;
    }

    return 0;
}
