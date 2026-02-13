#include <htm_gui/debugger.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>

#include <QApplication>
#include <QPalette>
#include <QStyleFactory>

#include "main_window.hpp"

namespace htm_gui {
namespace {

std::string normalize_theme(std::string theme) {
  std::transform(theme.begin(), theme.end(), theme.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return theme;
}

QPalette light_palette() {
  QPalette p;
  p.setColor(QPalette::Window, QColor(245, 245, 245));
  p.setColor(QPalette::WindowText, QColor(30, 30, 30));
  p.setColor(QPalette::Base, QColor(255, 255, 255));
  p.setColor(QPalette::AlternateBase, QColor(240, 240, 240));
  p.setColor(QPalette::ToolTipBase, QColor(255, 255, 220));
  p.setColor(QPalette::ToolTipText, QColor(20, 20, 20));
  p.setColor(QPalette::Text, QColor(25, 25, 25));
  p.setColor(QPalette::Button, QColor(236, 236, 236));
  p.setColor(QPalette::ButtonText, QColor(25, 25, 25));
  p.setColor(QPalette::BrightText, QColor(255, 255, 255));
  p.setColor(QPalette::Highlight, QColor(66, 133, 244));
  p.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
  return p;
}

QPalette dark_palette() {
  QPalette p;
  p.setColor(QPalette::Window, QColor(40, 44, 52));
  p.setColor(QPalette::WindowText, QColor(220, 220, 220));
  p.setColor(QPalette::Base, QColor(30, 34, 40));
  p.setColor(QPalette::AlternateBase, QColor(45, 50, 58));
  p.setColor(QPalette::ToolTipBase, QColor(60, 64, 72));
  p.setColor(QPalette::ToolTipText, QColor(230, 230, 230));
  p.setColor(QPalette::Text, QColor(220, 220, 220));
  p.setColor(QPalette::Button, QColor(55, 60, 68));
  p.setColor(QPalette::ButtonText, QColor(220, 220, 220));
  p.setColor(QPalette::BrightText, QColor(255, 100, 100));
  p.setColor(QPalette::Highlight, QColor(78, 120, 200));
  p.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
  return p;
}

void apply_theme(QApplication& app, const std::string& raw_theme) {
  if (raw_theme.empty()) {
    return;
  }

  const std::string theme = normalize_theme(raw_theme);
  if (theme != "light" && theme != "dark") {
    std::cerr << "Warning: unsupported GUI theme '" << raw_theme
              << "'. Expected 'light' or 'dark'. Using Qt default.\n";
    return;
  }

  app.setStyle(QStyleFactory::create("Fusion"));
  app.setPalette(theme == "dark" ? dark_palette() : light_palette());
}

}  // namespace

int run_debugger(int argc, char** argv, IHtmRuntime& runtime, const DebuggerOptions& opts) {
  QApplication app(argc, argv);
  apply_theme(app, opts.theme);

  htm_gui::qt::MainWindow window(runtime);
  if (!opts.window_title.empty()) {
    window.setWindowTitle(QString::fromStdString(opts.window_title));
  }

  window.show();
  return app.exec();
}

}  // namespace htm_gui
