#include <htm_gui/debugger.hpp>

#include <QApplication>

#include "main_window.hpp"

namespace htm_gui {

int run_debugger(int argc, char** argv, IHtmRuntime& runtime, const DebuggerOptions& opts) {
  QApplication app(argc, argv);

  htm_gui::qt::MainWindow window(runtime);
  if (!opts.window_title.empty()) {
    window.setWindowTitle(QString::fromStdString(opts.window_title));
  }

  window.show();
  return app.exec();
}

}  // namespace htm_gui
