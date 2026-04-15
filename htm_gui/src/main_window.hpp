#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <QMainWindow>
#include <QPointer>

#include <htm_gui/runtime.hpp>

class QDockWidget;
class QPlainTextEdit;
class QShowEvent;
class QString;

namespace htm_gui::qt {

class ImageView;

class MainWindow final : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(htm_gui::IHtmRuntime& runtime, QWidget* parent = nullptr);

public slots:
  void refresh();

protected:
  void showEvent(QShowEvent* event) override;

private slots:
  void stepOne();
  void stepN();
  void onColumnClicked(int x, int y);
  void onCellClicked(int x, int y);
  void showActiveCells();
  void showPredictCells();
  void showLearnCells();
  void markState();
  void pinCurrentProximal();
  void pinCurrentDistal();
  void saveLayout();
  void restoreLayout();
  void loadRuntimePatch();

private:
  enum class CellDisplayMode { Active, Predictive, Learning };
  struct PinnedDistalView {
    int col_x{-1};
    int col_y{-1};
    int cell{-1};
    int segment{-1};
    QPointer<QPlainTextEdit> text;
    QPointer<QDockWidget> dock;
  };
  struct PinnedProximalView {
    int col_x{-1};
    int col_y{-1};
    QPointer<QPlainTextEdit> text;
    QPointer<QDockWidget> dock;
  };

  QImage renderInput(const htm_gui::Snapshot& s, int max_w, int max_h) const;
  QImage renderColumns(const htm_gui::Snapshot& s, int max_w, int max_h) const;
  QImage renderCells(const htm_gui::Snapshot& s, int col_x, int col_y, int max_size) const;
  void updateProximalSynapsePanel();
  void updateDistalSynapsePanel();
  void updatePinnedProximalPanels();
  void updatePinnedDistalPanels();
  void applyInitialDockLayout();
  QString formatProximalSynapseText(const htm_gui::ProximalSynapseQuery& query, int col_x, int col_y) const;
  QString formatDistalSynapseText(const htm_gui::DistalSynapseQuery& query,
                                  int src_col_x,
                                  int src_col_y,
                                  int src_cell,
                                  int src_segment) const;
  static QString layoutFilePath();

  htm_gui::IHtmRuntime& runtime_;
  htm_gui::Snapshot snapshot_;

  int selected_col_x_{-1};
  int selected_col_y_{-1};
  int selected_cell_{-1};
  int selected_segment_{-1};
  int cells_side_{0};
  CellDisplayMode cell_mode_{CellDisplayMode::Active};
  std::optional<htm_gui::ProximalSynapseQuery> proximal_query_;
  std::optional<htm_gui::DistalSynapseQuery> distal_overlay_;
  std::vector<PinnedProximalView> pinned_proximal_views_;
  std::vector<PinnedDistalView> pinned_distal_views_;
  std::vector<QWidget*> marked_windows_;
  bool show_cell_overlay_{true};

  QWidget* sequence_widget_{nullptr};  // null when runtime doesn't support input selection
  QWidget* layer_widget_{nullptr};     // null when runtime doesn't support layer selection

  ImageView* input_view_{nullptr};
  ImageView* columns_view_{nullptr};
  ImageView* cells_view_{nullptr};
  QDockWidget* input_dock_{nullptr};
  QDockWidget* columns_dock_{nullptr};
  QDockWidget* cells_dock_{nullptr};
  QPlainTextEdit* proximal_text_{nullptr};
  QDockWidget* proximal_dock_{nullptr};
  QPlainTextEdit* distal_text_{nullptr};
  QDockWidget* distal_dock_{nullptr};
  bool initial_dock_layout_done_{false};
};

}  // namespace htm_gui::qt
