#include "pelican/core/PipelineApplication.h"
#include "LofarStreamDataClient.h"

#include "DDStream2.h"

#include "LofarTypes.h"
#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "LofarChunker.h"

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
        pApp.registerPipeline(new DDStream2);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();

    } catch (const QString& error) {
        std::cout << "Error in mainDDStream2.cpp : " << error.toStdString() << std::endl;
    }

    return 0;
}
