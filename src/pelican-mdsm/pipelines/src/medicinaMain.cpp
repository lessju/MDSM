#include "pelican/core/PipelineApplication.h"
#include "SpeadBeamDataClient.h"

#include "MedicinaMdsmPipeline.h"
#include "SpeadBeamAdapterTimeSeries.h"
#include "SpeadBeamChunker.h"

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
        pApp.registerPipeline(new MedicinaMdsmPipeline);

        // Set the data client.
        pApp.setDataClient("SpeadBeamDataClient");

        // Start the pipeline driver.
        pApp.start();

    } catch (const QString& error) {
        std::cout << "Error: " << error.toStdString() << std::endl;
    }

    return 0;
}
