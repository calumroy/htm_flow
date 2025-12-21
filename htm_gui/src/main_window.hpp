#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <QMainWindow>

#include <htm_gui/runtime.hpp>

namespace htm_gui::qt {

class ImageView;

class MainWindow final : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(htm_gui::IHtmRuntime& runtime, QWidget* parent = nullptr);

public slots:
  void refresh();

private slots:
  void stepOne();
  void stepN();
  void onColumnClicked(int x, int y);
  void onCellClicked(int x, int y);
  void showActiveCells();
  void showPredictCells();
  void showLearnCells();
  void markState();

private:
  enum class CellDisplayMode { Active, Predictive, Learning };

  QImage renderInput(const htm_gui::Snapshot& s, int max_w, int max_h) const;
  QImage renderColumns(const htm_gui::Snapshot& s, int max_w, int max_h) const;
  QImage renderCells(const htm_gui::Snapshot& s, int col_x, int col_y, int max_size) const;

  htm_gui::IHtmRuntime& runtime_;
  htm_gui::Snapshot snapshot_;

  int selected_col_x_{-1};
  int selected_col_y_{-1};
  int selected_cell_{-1};
  int selected_segment_{-1};
  int cells_side_{0};
  CellDisplayMode cell_mode_{CellDisplayMode::Active};
  std::optional<htm_gui::DistalSynapseQuery> distal_overlay_;
  std::vector<QWidget*> marked_windows_;
  bool show_cell_overlay_{true};

  ImageView* input_view_{nullptr};
  ImageView* columns_view_{nullptr};
  ImageView* cells_view_{nullptr};
};

}  // namespace htm_gui::qt
