#include "main_window.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include <QAction>
#include <QApplication>
#include <QDataStream>
#include <QDir>
#include <QDockWidget>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QImage>
#include <QInputDialog>
#include <QKeySequence>
#include <QLabel>
#include <QComboBox>
#include <QPainter>
#include <QPlainTextEdit>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QToolBar>
#include <QTimer>
#include <QShowEvent>
#include <QWidget>

#include "views.hpp"

namespace {

QColor red() { return QColor(0xFF, 0x00, 0x00); }
QColor green() { return QColor(0x00, 0xFF, 0x00); }
QColor blue() { return QColor(0x40, 0x30, 0xFF); }
QColor black() { return QColor(0x00, 0x00, 0x00); }
QColor dark_green() { return QColor(0x00, 0x80, 0x40); }
QColor yellow() { return QColor(0xFF, 0xD0, 0x00); }
QColor light_red() { return QColor(0xFF, 0x60, 0x60); }
QColor light_green() { return QColor(0x00, 0xFF, 0x00, 0x80); }
QColor transp_blue() { return QColor(0x00, 0x00, 0xFF, 0x30); }

QImage renderColumnsWithCellOverlay(const htm_gui::Snapshot& s,
                                   int max_w,
                                   int max_h,
                                   int sel_x,
                                   int sel_y,
                                   const std::optional<htm_gui::DistalSynapseQuery>* distal_overlay,
                                   bool show_cell_overlay) {
  const int src_w = s.columns_shape.cols;
  const int src_h = s.columns_shape.rows;

  if (src_w <= 0 || src_h <= 0) {
    QImage img(1, 1, QImage::Format_ARGB32);
    img.fill(red());
    return img;
  }

  // If the logical grid is bigger than our max output, fall back to the legacy downsample path.
  if (!show_cell_overlay || src_w > max_w || src_h > max_h) {
    const int out_w = std::max(1, std::min(max_w, src_w));
    const int out_h = std::max(1, std::min(max_h, src_h));

    std::unordered_set<int> active;
    active.reserve(s.active_column_indices.size());
    for (int idx : s.active_column_indices) active.insert(idx);

    QImage img(out_w, out_h, QImage::Format_ARGB32);
    for (int oy = 0; oy < out_h; ++oy) {
      const int sy = (oy * src_h) / out_h;
      for (int ox = 0; ox < out_w; ++ox) {
        const int sx = (ox * src_w) / out_w;
        const int idx = htm_gui::flatten_xy(sx, sy, src_w);
        img.setPixelColor(ox, oy, active.count(idx) ? green() : red());
      }
    }

    // Distal overlay targets (connected) as single pixels.
    if (distal_overlay && distal_overlay->has_value()) {
      for (const auto& syn : distal_overlay->value().synapses) {
        if (syn.dst_column_x < 0 || syn.dst_column_x >= src_w || syn.dst_column_y < 0 || syn.dst_column_y >= src_h) {
          continue;
        }
        const int px = (syn.dst_column_x * out_w) / src_w;
        const int py = (syn.dst_column_y * out_h) / src_h;
        if (px < 0 || px >= out_w || py < 0 || py >= out_h) {
          continue;
        }
        if (syn.connected) {
          img.setPixelColor(px, py, blue());
        }
      }
    }

    // Selected column highlight as a cross.
    if (sel_x >= 0 && sel_y >= 0) {
      const int px = (sel_x * out_w) / src_w;
      const int py = (sel_y * out_h) / src_h;
      for (int dx = -2; dx <= 2; ++dx) {
        const int x = px + dx;
        if (x >= 0 && x < out_w && py >= 0 && py < out_h) img.setPixelColor(x, py, yellow());
      }
      for (int dy = -2; dy <= 2; ++dy) {
        const int y = py + dy;
        if (px >= 0 && px < out_w && y >= 0 && y < out_h) img.setPixelColor(px, y, yellow());
      }
    }

    return img;
  }

  // Tile rendering: each logical column becomes a tile of `tile_px` pixels.
  const int tile_px = std::max(2, std::min(max_w / src_w, max_h / src_h));
  const int out_w = std::max(1, src_w * tile_px);
  const int out_h = std::max(1, src_h * tile_px);

  std::unordered_set<int> active;
  active.reserve(s.active_column_indices.size());
  for (int idx : s.active_column_indices) active.insert(idx);

  QImage img(out_w, out_h, QImage::Format_ARGB32);
  img.fill(QColor(0, 0, 0, 0));

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, false);
  painter.setPen(Qt::NoPen);

  const int cells = s.cells_per_column;
  const bool can_draw_cells = (cells > 0 && cells <= 64 && int(s.column_cell_masks.size()) >= src_w * src_h);
  const int side = can_draw_cells ? int(std::ceil(std::sqrt(double(cells)))) : 0;

  std::vector<std::uint64_t> distal_cell_masks;
  if (can_draw_cells && distal_overlay && distal_overlay->has_value()) {
    distal_cell_masks.assign(static_cast<std::size_t>(src_w * src_h), 0);
    for (const auto& syn : distal_overlay->value().synapses) {
      if (!syn.connected) {
        continue;
      }
      if (syn.dst_column_x < 0 || syn.dst_column_x >= src_w || syn.dst_column_y < 0 || syn.dst_column_y >= src_h ||
          syn.dst_cell < 0 || syn.dst_cell >= cells) {
        continue;
      }
      const int idx = htm_gui::flatten_xy(syn.dst_column_x, syn.dst_column_y, src_w);
      distal_cell_masks[static_cast<std::size_t>(idx)] |= (std::uint64_t(1) << syn.dst_cell);
    }
  }

  for (int sy = 0; sy < src_h; ++sy) {
    for (int sx = 0; sx < src_w; ++sx) {
      const int col_idx = htm_gui::flatten_xy(sx, sy, src_w);
      const int x0 = sx * tile_px;
      const int y0 = sy * tile_px;

      painter.setBrush(active.count(col_idx) ? green() : red());
      painter.drawRect(x0, y0, tile_px, tile_px);

      if (!can_draw_cells) {
        continue;
      }

      const auto masks = s.column_cell_masks[static_cast<std::size_t>(col_idx)];
      const std::uint64_t distal_mask =
          (col_idx >= 0 && col_idx < int(distal_cell_masks.size())) ? distal_cell_masks[static_cast<std::size_t>(col_idx)]
                                                                     : 0;

      // Inner area for cell glyphs.
      const int pad = 1;
      const int inner = std::max(0, tile_px - 2 * pad);
      if (inner <= 0) {
        continue;
      }

      for (int c = 0; c < cells; ++c) {
        const int gx = c % side;
        const int gy = c / side;

        // Distribute pixels evenly across the inner area.
        const int cx0 = x0 + pad + (gx * inner) / side;
        const int cy0 = y0 + pad + (gy * inner) / side;
        const int cx1 = x0 + pad + ((gx + 1) * inner) / side;
        const int cy1 = y0 + pad + ((gy + 1) * inner) / side;

        const int cw = std::max(1, cx1 - cx0);
        const int ch = std::max(1, cy1 - cy0);

        const std::uint64_t bit = (std::uint64_t(1) << c);
        const bool is_pred = (masks.predictive & bit) != 0;
        const bool is_act = (masks.active & bit) != 0;
        const bool is_learn = (masks.learning & bit) != 0;
        const bool is_distal_target = (distal_mask & bit) != 0;

        // Predictive: black outer square (dominant signal).
        if (is_pred) {
          painter.setBrush(black());
          painter.drawRect(cx0, cy0, cw, ch);
        }

        // Active: blue inner square (can overlay predictive and show active even when not predictive).
        if (is_act) {
          const int inset = std::max(1, std::min(cw, ch) / 4);
          const int ix0 = cx0 + inset;
          const int iy0 = cy0 + inset;
          const int iw = std::max(1, cw - 2 * inset);
          const int ih = std::max(1, ch - 2 * inset);
          painter.setBrush(blue());
          painter.drawRect(ix0, iy0, iw, ih);
        }

        // Learning: dark-green dot in the center.
        if (is_learn) {
          const int dot = std::max(1, std::min(cw, ch) / 3);
          const int dx0 = cx0 + (cw - dot) / 2;
          const int dy0 = cy0 + (ch - dot) / 2;
          painter.setBrush(dark_green());
          painter.drawRect(dx0, dy0, dot, dot);
        }

        if (is_distal_target) {
          QPen pen(blue());
          pen.setWidth(1);
          painter.setPen(pen);
          painter.setBrush(Qt::NoBrush);
          painter.drawRect(cx0, cy0, std::max(1, cw - 1), std::max(1, ch - 1));
          painter.setPen(Qt::NoPen);
        }
      }
    }
  }

  painter.setBrush(Qt::NoBrush);

  // Distal overlay targets: draw a thin blue outline around the destination column tile.
  if (distal_overlay && distal_overlay->has_value()) {
    QPen pen(blue());
    pen.setWidth(2);
    painter.setPen(pen);
    for (const auto& syn : distal_overlay->value().synapses) {
      if (!syn.connected) {
        continue;
      }
      if (syn.dst_column_x < 0 || syn.dst_column_x >= src_w || syn.dst_column_y < 0 || syn.dst_column_y >= src_h) {
        continue;
      }
      const int x0 = syn.dst_column_x * tile_px;
      const int y0 = syn.dst_column_y * tile_px;
      painter.drawRect(x0 + 1, y0 + 1, tile_px - 2, tile_px - 2);
    }
  }

  // Selected column highlight: thick yellow outline.
  if (sel_x >= 0 && sel_y >= 0) {
    QPen pen(yellow());
    pen.setWidth(3);
    painter.setPen(pen);
    const int x0 = sel_x * tile_px;
    const int y0 = sel_y * tile_px;
    painter.drawRect(x0 + 1, y0 + 1, tile_px - 2, tile_px - 2);
  }

  return img;
}

class FrozenWindow final : public QMainWindow {
public:
  explicit FrozenWindow(htm_gui::Snapshot snapshot, QWidget* parent = nullptr) : QMainWindow(parent), snapshot_(std::move(snapshot)) {
    setWindowTitle(QString("Marked state (t=%1)").arg(snapshot_.timestep));

    input_view_ = new htm_gui::qt::ImageView(this);
    columns_view_ = new htm_gui::qt::ImageView(this);
    cells_view_ = new htm_gui::qt::ImageView(this);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setObjectName("FrozenMainSplitter");
    splitter->setChildrenCollapsible(false);
    splitter->setOpaqueResize(true);
    splitter->addWidget(input_view_);
    splitter->addWidget(columns_view_);
    splitter->addWidget(cells_view_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    splitter->setStretchFactor(2, 0);

    setCentralWidget(splitter);
    statusBar()->showMessage("Marked (static)");

    input_view_->setLogicalGrid(snapshot_.input_shape.cols, snapshot_.input_shape.rows);
    columns_view_->setLogicalGrid(snapshot_.columns_shape.cols, snapshot_.columns_shape.rows);

    connect(columns_view_, &htm_gui::qt::ImageView::clicked, this, [this](int x, int y) {
      selected_col_x_ = x;
      selected_col_y_ = y;
      redraw();
    });

    redraw();
  }

private:
  static QImage renderInputRaw(const htm_gui::Snapshot& s, int max_w, int max_h) {
    const int src_w = s.input_shape.cols;
    const int src_h = s.input_shape.rows;

    const int out_w = std::max(1, std::min(max_w, src_w));
    const int out_h = std::max(1, std::min(max_h, src_h));

    QImage img(out_w, out_h, QImage::Format_ARGB32);
    if (!s.input) {
      img.fill(red());
      return img;
    }

    for (int oy = 0; oy < out_h; ++oy) {
      const int sy = (oy * src_h) / out_h;
      for (int ox = 0; ox < out_w; ++ox) {
        const int sx = (ox * src_w) / out_w;
        const int v = (*s.input)[sy * src_w + sx];
        img.setPixelColor(ox, oy, v ? green() : red());
      }
    }
    return img;
  }

  static QImage renderColumnsRaw(const htm_gui::Snapshot& s, int max_w, int max_h, int sel_x, int sel_y) {
    return renderColumnsWithCellOverlay(s, max_w, max_h, sel_x, sel_y, nullptr, true);
  }

  static QImage renderCellsRaw(const htm_gui::Snapshot& s, int col_x, int col_y, int max_size) {
    const int cells = s.cells_per_column;
    if (cells <= 0 || cells > 64) {
      QImage img(1, 1, QImage::Format_ARGB32);
      img.fill(black());
      return img;
    }

    const int col_idx = htm_gui::flatten_xy(col_x, col_y, s.columns_shape.cols);
    if (col_idx < 0 || col_idx >= int(s.column_cell_masks.size())) {
      QImage img(1, 1, QImage::Format_ARGB32);
      img.fill(black());
      return img;
    }

    const auto masks = s.column_cell_masks[col_idx];
    const int side = int(std::ceil(std::sqrt(double(cells))));
    const int out = std::max(1, std::min(max_size, side * 12));
    QImage img(out, out, QImage::Format_ARGB32);
    img.fill(QColor(0, 0, 0, 0));

    const int cell_px = std::max(1, out / side);
    for (int c = 0; c < cells; ++c) {
      const int gx = c % side;
      const int gy = c / side;

      QColor color = QColor(0, 0, 0xFF, 0x30);
      if (masks.learning & (std::uint64_t(1) << c)) {
        color = dark_green();
      } else if (masks.predictive & (std::uint64_t(1) << c)) {
        color = black();
      } else if (masks.active & (std::uint64_t(1) << c)) {
        color = blue();
      }

      for (int py = gy * cell_px; py < std::min(out, (gy + 1) * cell_px); ++py) {
        for (int px = gx * cell_px; px < std::min(out, (gx + 1) * cell_px); ++px) {
          img.setPixelColor(px, py, color);
        }
      }
    }

    return img;
  }

  void redraw() {
    input_view_->setImage(renderInputRaw(snapshot_, 512, 512));
    columns_view_->setImage(renderColumnsRaw(snapshot_, 512, 512, selected_col_x_, selected_col_y_));

    if (selected_col_x_ >= 0 && selected_col_y_ >= 0) {
      const int side = int(std::ceil(std::sqrt(double(snapshot_.cells_per_column))));
      cells_view_->setLogicalGrid(side, side);
      cells_view_->setImage(renderCellsRaw(snapshot_, selected_col_x_, selected_col_y_, 256));
    } else {
      cells_view_->setImage(QImage());
    }
  }

  htm_gui::Snapshot snapshot_;
  int selected_col_x_{-1};
  int selected_col_y_{-1};

  htm_gui::qt::ImageView* input_view_{nullptr};
  htm_gui::qt::ImageView* columns_view_{nullptr};
  htm_gui::qt::ImageView* cells_view_{nullptr};
};

}  // namespace

namespace htm_gui::qt {

namespace {

ImageView* focusedImageView() {
  QWidget* w = QApplication::focusWidget();
  while (w) {
    if (auto* v = dynamic_cast<ImageView*>(w)) {
      return v;
    }
    w = w->parentWidget();
  }
  return nullptr;
}

bool distalSynapseLess(const htm_gui::DistalSynapseInfo& a, const htm_gui::DistalSynapseInfo& b) {
  if (a.dst_column_y != b.dst_column_y) {
    return a.dst_column_y < b.dst_column_y;
  }
  if (a.dst_column_x != b.dst_column_x) {
    return a.dst_column_x < b.dst_column_x;
  }
  if (a.dst_cell != b.dst_cell) {
    return a.dst_cell < b.dst_cell;
  }
  if (a.connected != b.connected) {
    return a.connected && !b.connected;
  }
  return a.permanence > b.permanence;
}

}  // namespace

MainWindow::MainWindow(htm_gui::IHtmRuntime& runtime, QWidget* parent)
    : QMainWindow(parent), runtime_(runtime) {
  setWindowTitle(QString::fromStdString(runtime_.name()));

  input_view_ = new ImageView(this);
  columns_view_ = new ImageView(this);
  cells_view_ = new ImageView(this);

  setDockNestingEnabled(true);

  // Main visual panes as docks so they can be drag-reordered interactively.
  input_dock_ = new QDockWidget("Input", this);
  input_dock_->setObjectName("InputViewDock");
  input_dock_->setAllowedAreas(Qt::AllDockWidgetAreas);
  input_dock_->setWidget(input_view_);
  addDockWidget(Qt::LeftDockWidgetArea, input_dock_);

  columns_dock_ = new QDockWidget("Columns", this);
  columns_dock_->setObjectName("ColumnsViewDock");
  columns_dock_->setAllowedAreas(Qt::AllDockWidgetAreas);
  columns_dock_->setWidget(columns_view_);
  addDockWidget(Qt::LeftDockWidgetArea, columns_dock_);
  splitDockWidget(input_dock_, columns_dock_, Qt::Horizontal);

  cells_dock_ = new QDockWidget("Cells", this);
  cells_dock_->setObjectName("CellsViewDock");
  cells_dock_->setAllowedAreas(Qt::AllDockWidgetAreas);
  cells_dock_->setWidget(cells_view_);
  addDockWidget(Qt::LeftDockWidgetArea, cells_dock_);
  splitDockWidget(columns_dock_, cells_dock_, Qt::Horizontal);

  // Distal synapse panel: dockable + floatable so you can move/resize it independently.
  distal_text_ = new QPlainTextEdit(this);
  distal_text_->setReadOnly(true);
  distal_text_->setPlaceholderText("Select a column, then click a cell and choose a segment to view distal synapses.");

  distal_dock_ = new QDockWidget("Distal synapses", this);
  distal_dock_->setObjectName("DistalSynapsesDock");
  distal_dock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
  distal_dock_->setWidget(distal_text_);
  addDockWidget(Qt::RightDockWidgetArea, distal_dock_);

  // Proximal synapse panel: dockable + floatable (mirrors distal panel).
  proximal_text_ = new QPlainTextEdit(this);
  proximal_text_->setReadOnly(true);
  proximal_text_->setPlaceholderText("Select a column to view its proximal (potential) synapses.");

  proximal_dock_ = new QDockWidget("Proximal synapses", this);
  proximal_dock_->setObjectName("ProximalSynapsesDock");
  proximal_dock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
  proximal_dock_->setWidget(proximal_text_);
  addDockWidget(Qt::RightDockWidgetArea, proximal_dock_);

  auto* tb = addToolBar("Controls");
  auto* step_one = new QAction("Step", this);
  auto* step_n = new QAction("N steps", this);
  auto* show_active = new QAction("Active Cells", this);
  auto* show_pred = new QAction("Predict Cells", this);
  auto* show_learn = new QAction("Learn Cells", this);
  auto* toggle_overlay = new QAction("Cell overlay", this);
  auto* zoom_in = new QAction("Zoom In", this);
  auto* zoom_out = new QAction("Zoom Out", this);
  auto* mark = new QAction("Mark", this);
  auto* pin_proximal = new QAction("Pin Proximal", this);
  auto* pin_distal = new QAction("Pin Distal", this);

  // Space bar steps the simulation (global within the app window).
  step_one->setShortcut(QKeySequence(Qt::Key_Space));
  step_one->setShortcutContext(Qt::ApplicationShortcut);

  tb->addAction(step_one);
  tb->addAction(step_n);

  // Optional input sequence dropdown. Only shown when the runtime advertises sequences.
  const auto seqs = runtime_.input_sequences();
  if (!seqs.empty()) {
    tb->addSeparator();

    auto* label = new QLabel("Input:", this);
    tb->addWidget(label);

    auto* combo = new QComboBox(this);
    combo->setToolTip("Select the input stimulus sequence (runtime-dependent).");
    combo->setMinimumContentsLength(18);

    {
      const QSignalBlocker blocker(combo);
      for (const auto& s : seqs) {
        combo->addItem(QString::fromStdString(s.name), QVariant(s.id));
      }

      // Set current selection to the runtime's current sequence if present.
      const int cur_id = runtime_.input_sequence();
      int found = -1;
      for (int i = 0; i < combo->count(); ++i) {
        if (combo->itemData(i).toInt() == cur_id) {
          found = i;
          break;
        }
      }
      combo->setCurrentIndex(found >= 0 ? found : 0);
    }

    connect(combo, &QComboBox::currentIndexChanged, this, [this, combo](int idx) {
      if (idx < 0) {
        return;
      }
      const int id = combo->itemData(idx).toInt();
      runtime_.set_input_sequence(id);
      refresh();
    });

    auto* w = new QWidget(this);
    auto* layout = new QHBoxLayout(w);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(combo);
    w->setLayout(layout);
    tb->addWidget(w);
    sequence_widget_ = w;
  }

  // Optional layer selector dropdown. Only shown when the runtime has multiple layers.
  const auto layer_opts = runtime_.layer_options();
  if (layer_opts.size() > 1) {
    tb->addSeparator();

    auto* layer_label = new QLabel("Layer:", this);
    tb->addWidget(layer_label);

    auto* layer_combo = new QComboBox(this);
    layer_combo->setToolTip("Select which layer to visualize in a multi-layer region.");
    layer_combo->setMinimumContentsLength(10);

    {
      const QSignalBlocker blocker(layer_combo);
      for (const auto& opt : layer_opts) {
        layer_combo->addItem(QString::fromStdString(opt.name), QVariant(opt.id));
      }

      // Set current selection to the runtime's active layer.
      const int cur_layer = runtime_.active_layer();
      int found = -1;
      for (int i = 0; i < layer_combo->count(); ++i) {
        if (layer_combo->itemData(i).toInt() == cur_layer) {
          found = i;
          break;
        }
      }
      layer_combo->setCurrentIndex(found >= 0 ? found : 0);
    }

    connect(layer_combo, &QComboBox::currentIndexChanged, this, [this, layer_combo](int idx) {
      if (idx < 0) {
        return;
      }
      const int layer_idx = layer_combo->itemData(idx).toInt();
      runtime_.set_active_layer(layer_idx);
      refresh();
    });

    auto* layer_w = new QWidget(this);
    auto* layer_layout = new QHBoxLayout(layer_w);
    layer_layout->setContentsMargins(0, 0, 0, 0);
    layer_layout->addWidget(layer_combo);
    layer_w->setLayout(layer_layout);
    tb->addWidget(layer_w);
    layer_widget_ = layer_w;
  }

  tb->addSeparator();
  tb->addAction(show_active);
  tb->addAction(show_pred);
  tb->addAction(show_learn);
  tb->addAction(toggle_overlay);
  tb->addSeparator();
  tb->addAction(zoom_in);
  tb->addAction(zoom_out);
  auto* save_layout = new QAction("Save Layout", this);
  auto* restore_layout = new QAction("Restore Layout", this);

  tb->addSeparator();
  tb->addAction(mark);
  tb->addAction(pin_proximal);
  tb->addAction(pin_distal);
  tb->addSeparator();
  tb->addAction(save_layout);
  tb->addAction(restore_layout);

  connect(step_one, &QAction::triggered, this, &MainWindow::stepOne);
  connect(step_n, &QAction::triggered, this, &MainWindow::stepN);
  connect(show_active, &QAction::triggered, this, &MainWindow::showActiveCells);
  connect(show_pred, &QAction::triggered, this, &MainWindow::showPredictCells);
  connect(show_learn, &QAction::triggered, this, &MainWindow::showLearnCells);
  connect(mark, &QAction::triggered, this, &MainWindow::markState);
  connect(pin_proximal, &QAction::triggered, this, &MainWindow::pinCurrentProximal);
  connect(pin_distal, &QAction::triggered, this, &MainWindow::pinCurrentDistal);
  connect(save_layout, &QAction::triggered, this, &MainWindow::saveLayout);
  connect(restore_layout, &QAction::triggered, this, &MainWindow::restoreLayout);

  // Global zoom shortcuts apply to the currently focused ImageView (input / columns / cells).
  zoom_in->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Plus));
  zoom_in->setShortcutContext(Qt::ApplicationShortcut);
  zoom_out->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Minus));
  zoom_out->setShortcutContext(Qt::ApplicationShortcut);

  connect(zoom_in, &QAction::triggered, this, [this]() {
    if (auto* v = focusedImageView()) {
      v->zoomIn();
    }
  });
  connect(zoom_out, &QAction::triggered, this, [this]() {
    if (auto* v = focusedImageView()) {
      v->zoomOut();
    }
  });

  // Handle the common Ctrl+= for Ctrl++ on many keyboards.
  auto* zoom_in_alt = new QAction(this);
  zoom_in_alt->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Equal));
  zoom_in_alt->setShortcutContext(Qt::ApplicationShortcut);
  addAction(zoom_in_alt);
  connect(zoom_in_alt, &QAction::triggered, this, [this]() {
    if (auto* v = focusedImageView()) {
      v->zoomIn();
    }
  });

  toggle_overlay->setCheckable(true);
  toggle_overlay->setChecked(show_cell_overlay_);
  connect(toggle_overlay, &QAction::toggled, this, [this](bool checked) {
    show_cell_overlay_ = checked;
    refresh();
  });

  connect(columns_view_, &ImageView::clicked, this, &MainWindow::onColumnClicked);
  connect(cells_view_, &ImageView::clicked, this, &MainWindow::onCellClicked);

  auto* legend = new QLabel(this);
  legend->setText(
      "Columns: green=active red=inactive yellow=selected blue=distal target  |  Cells: black=predictive blue=active dark green=learning");
  legend->setToolTip("Per-column overlay: black outer=pred, blue inner=active, dark-green dot=learning.");
  statusBar()->addPermanentWidget(legend);

  // Keep proximal/distal as tabs so the main image panes get most of the width on startup.
  tabifyDockWidget(proximal_dock_, distal_dock_);
  distal_dock_->raise();

  statusBar()->showMessage("Ready");
  refresh();
}

void MainWindow::showEvent(QShowEvent* event) {
  QMainWindow::showEvent(event);
  if (initial_dock_layout_done_) {
    return;
  }
  initial_dock_layout_done_ = true;

  QTimer::singleShot(0, this, [this]() {
    if (QFile::exists(layoutFilePath())) {
      restoreLayout();
    } else {
      applyInitialDockLayout();
      QTimer::singleShot(0, this, [this]() {
        applyInitialDockLayout();
      });
    }
  });
}

void MainWindow::applyInitialDockLayout() {
  if (!input_dock_ || !columns_dock_ || !cells_dock_ || !proximal_dock_) {
    return;
  }

  // Keep the right-side synapse tabs relatively narrow, then split remaining width evenly across main panes.
  resizeDocks({input_dock_, columns_dock_, cells_dock_, proximal_dock_}, {3, 3, 3, 1}, Qt::Horizontal);
  resizeDocks({input_dock_, columns_dock_, cells_dock_}, {1, 1, 1}, Qt::Horizontal);
}

void MainWindow::refresh() {
  snapshot_ = runtime_.snapshot();

  if (snapshot_.input_shape.rows <= 0 || snapshot_.input_shape.cols <= 0 || snapshot_.columns_shape.rows <= 0 ||
      snapshot_.columns_shape.cols <= 0) {
    statusBar()->showMessage("Invalid snapshot shapes");
    return;
  }

  // Keep selection consistent with the new snapshot, and refresh the distal synapse query so the panel stays up-to-date
  // across step updates.
  const int grid_w = snapshot_.columns_shape.cols;
  const int grid_h = snapshot_.columns_shape.rows;

  if (selected_col_x_ < 0 || selected_col_y_ < 0 || selected_col_x_ >= grid_w || selected_col_y_ >= grid_h) {
    selected_col_x_ = -1;
    selected_col_y_ = -1;
    selected_cell_ = -1;
    selected_segment_ = -1;
    proximal_query_.reset();
    distal_overlay_.reset();
  }

  if (selected_cell_ >= 0 && selected_cell_ >= snapshot_.cells_per_column) {
    selected_cell_ = -1;
    selected_segment_ = -1;
    distal_overlay_.reset();
  }

  if (selected_col_x_ >= 0 && selected_col_y_ >= 0) {
    proximal_query_ = runtime_.query_proximal(selected_col_x_, selected_col_y_);
  } else {
    proximal_query_.reset();
  }

  if (selected_col_x_ >= 0 && selected_col_y_ >= 0 && selected_cell_ >= 0 && selected_segment_ >= 0) {
    const int nsegs = runtime_.num_segments(selected_col_x_, selected_col_y_, selected_cell_);
    if (nsegs <= 0 || selected_segment_ >= nsegs) {
      selected_segment_ = -1;
      distal_overlay_.reset();
    } else {
      distal_overlay_ = runtime_.query_distal(selected_col_x_, selected_col_y_, selected_cell_, selected_segment_);
    }
  }

  input_view_->setLogicalGrid(snapshot_.input_shape.cols, snapshot_.input_shape.rows);
  columns_view_->setLogicalGrid(snapshot_.columns_shape.cols, snapshot_.columns_shape.rows);

  input_view_->setImage(renderInput(snapshot_, 512, 512));
  columns_view_->setImage(renderColumns(snapshot_, 512, 512));

  if (selected_col_x_ >= 0 && selected_col_y_ >= 0) {
    cells_side_ = int(std::ceil(std::sqrt(double(snapshot_.cells_per_column))));
    cells_view_->setLogicalGrid(cells_side_, cells_side_);
    cells_view_->setImage(renderCells(snapshot_, selected_col_x_, selected_col_y_, 256));
  } else {
    cells_view_->setImage(QImage());
  }

  updateProximalSynapsePanel();
  updateDistalSynapsePanel();
  updatePinnedProximalPanels();
  updatePinnedDistalPanels();

  std::string msg = "t=" + std::to_string(snapshot_.timestep) +
                    " active_cols=" + std::to_string(snapshot_.active_column_indices.size()) +
                    " grid=" + std::to_string(snapshot_.columns_shape.cols) + "x" +
                    std::to_string(snapshot_.columns_shape.rows);
  // Add layer info if this is a multi-layer runtime
  const int num_layers = runtime_.num_layers();
  if (num_layers > 1) {
    msg = "Layer " + std::to_string(runtime_.active_layer()) + "/" + std::to_string(num_layers) + "  " + msg;
  }
  statusBar()->showMessage(QString::fromStdString(msg));
}

void MainWindow::stepOne() {
  runtime_.step(1);
  refresh();
}

void MainWindow::stepN() {
  bool ok = false;
  const int n = QInputDialog::getInt(this, "number of steps", "steps:", 10, 1, 1'000'000, 1, &ok);
  if (!ok) {
    return;
  }
  runtime_.step(n);
  refresh();
}

void MainWindow::onColumnClicked(int x, int y) {
  selected_col_x_ = x;
  selected_col_y_ = y;
  selected_cell_ = -1;
  selected_segment_ = -1;
  proximal_query_.reset();
  distal_overlay_.reset();
  refresh();
}

void MainWindow::onCellClicked(int x, int y) {
  if (selected_col_x_ < 0 || selected_col_y_ < 0 || snapshot_.cells_per_column <= 0) {
    return;
  }

  const int cell = y * cells_side_ + x;
  if (cell < 0 || cell >= snapshot_.cells_per_column) {
    return;
  }

  selected_cell_ = cell;

  const int nsegs = runtime_.num_segments(selected_col_x_, selected_col_y_, selected_cell_);
  if (nsegs <= 0) {
    selected_segment_ = -1;
    distal_overlay_.reset();
    refresh();
    return;
  }

  bool ok = false;
  const int seg = QInputDialog::getInt(this, "segment", "segment index:", 0, 0, nsegs - 1, 1, &ok);
  if (!ok) {
    return;
  }

  selected_segment_ = seg;
  distal_overlay_ = runtime_.query_distal(selected_col_x_, selected_col_y_, selected_cell_, selected_segment_);
  refresh();
}

void MainWindow::showActiveCells() {
  cell_mode_ = CellDisplayMode::Active;
  refresh();
}

void MainWindow::showPredictCells() {
  cell_mode_ = CellDisplayMode::Predictive;
  refresh();
}

void MainWindow::showLearnCells() {
  cell_mode_ = CellDisplayMode::Learning;
  refresh();
}

void MainWindow::updateProximalSynapsePanel() {
  if (!proximal_text_) {
    return;
  }

  if (selected_col_x_ < 0 || selected_col_y_ < 0) {
    proximal_text_->setPlainText("No column selected.\n\nClick a column to view its proximal synapses.");
    return;
  }

  if (!proximal_query_.has_value()) {
    proximal_query_ = runtime_.query_proximal(selected_col_x_, selected_col_y_);
  }

  proximal_text_->setPlainText(formatProximalSynapseText(*proximal_query_, selected_col_x_, selected_col_y_));
}

QString MainWindow::formatProximalSynapseText(const htm_gui::ProximalSynapseQuery& query, int col_x, int col_y) const {
  auto syns = query.synapses;
  std::sort(syns.begin(), syns.end(), [](const htm_gui::ProximalSynapseInfo& a, const htm_gui::ProximalSynapseInfo& b) {
    return a.permanence > b.permanence;
  });

  int connected_count = 0;
  int input_on_count = 0;
  for (const auto& s : syns) {
    if (s.connected) ++connected_count;
    if (s.input_value) ++input_on_count;
  }

  QString out;
  out += QString("Column (%1,%2)\n").arg(col_x).arg(col_y);
  out += QString("Overlap: %1   Potential overlap: %2\n")
             .arg(QString::number(query.overlap, 'f', 6))
             .arg(QString::number(query.potential_overlap, 'f', 6));
  out += QString("Synapses: %1  connected: %2  input_on: %3\n\n")
             .arg(int(syns.size()))
             .arg(connected_count)
             .arg(input_on_count);

  // Legend:
  // - conn: runtime-provided connected flag (perm >= connected threshold)
  // - input: current input bit at that location
  int i = 0;
  for (const auto& s : syns) {
    out += QString("%1) in=(%2,%3)  perm=%4  conn=%5  input=%6\n")
               .arg(i++, 3)
               .arg(s.input_x, 4)
               .arg(s.input_y, 4)
               .arg(QString::number(s.permanence, 'f', 4))
               .arg(s.connected ? "Y" : "N")
               .arg(s.input_value ? "1" : "0");
  }
  return out;
}

void MainWindow::updateDistalSynapsePanel() {
  if (!distal_text_) {
    return;
  }

  if (selected_col_x_ < 0 || selected_col_y_ < 0) {
    distal_text_->setPlainText("No column selected.\n\nClick a column, then click a cell and choose a segment.");
    return;
  }

  if (!distal_overlay_.has_value() || selected_cell_ < 0 || selected_segment_ < 0) {
    distal_text_->setPlainText(QString("Column (%1,%2)\n\nSelect a cell + segment to view distal synapses.")
                                   .arg(selected_col_x_)
                                   .arg(selected_col_y_));
    return;
  }

  distal_text_->setPlainText(
      formatDistalSynapseText(*distal_overlay_, selected_col_x_, selected_col_y_, selected_cell_, selected_segment_));
}

QString MainWindow::formatDistalSynapseText(const htm_gui::DistalSynapseQuery& query,
                                            int src_col_x,
                                            int src_col_y,
                                            int src_cell,
                                            int src_segment) const {
  auto syns = query.synapses;
  std::sort(syns.begin(), syns.end(), distalSynapseLess);

  int connected_count = 0;
  for (const auto& s : syns) {
    if (s.connected) {
      ++connected_count;
    }
  }

  QString syn_lines;
  const int grid_w = snapshot_.columns_shape.cols;
  const int grid_h = snapshot_.columns_shape.rows;

  bool cell_predictive = false;
  bool cell_active = false;
  bool cell_learning = false;
  if (src_cell >= 0 && src_cell < 64 && src_col_x >= 0 && src_col_y >= 0 && grid_w > 0) {
    const int src_col_idx = htm_gui::flatten_xy(src_col_x, src_col_y, grid_w);
    if (src_col_idx >= 0 && src_col_idx < int(snapshot_.column_cell_masks.size())) {
      const auto masks = snapshot_.column_cell_masks[src_col_idx];
      cell_predictive = (masks.predictive & (std::uint64_t(1) << src_cell)) != 0;
      cell_active = (masks.active & (std::uint64_t(1) << src_cell)) != 0;
      cell_learning = (masks.learning & (std::uint64_t(1) << src_cell)) != 0;
    }
  }

  const int threshold = runtime_.activation_threshold();
  int active_connected = 0;
  int tgt_active_total = 0;
  int i = 0;
  for (const auto& s : syns) {
    bool dst_active = false;
    if (s.dst_column_x >= 0 && s.dst_column_x < grid_w && s.dst_column_y >= 0 && s.dst_column_y < grid_h && s.dst_cell >= 0 &&
        s.dst_cell < 64) {
      const int col_idx = htm_gui::flatten_xy(s.dst_column_x, s.dst_column_y, grid_w);
      if (col_idx >= 0 && col_idx < int(snapshot_.column_cell_masks.size())) {
        const auto masks = snapshot_.column_cell_masks[col_idx];
        dst_active = (masks.active & (std::uint64_t(1) << s.dst_cell)) != 0;
      }
    }
    if (dst_active) {
      ++tgt_active_total;
    }
    if (s.connected && dst_active) {
      ++active_connected;
    }

    syn_lines += QString("%1) dst=(%2,%3) cell=%4  perm=%5  conn=%6  tgt_active=%7\n")
                     .arg(i++, 2)
                     .arg(s.dst_column_x, 4)
                     .arg(s.dst_column_y, 4)
                     .arg(s.dst_cell, 2)
                     .arg(QString::number(s.permanence, 'f', 4))
                     .arg(s.connected ? "Y" : "N")
                     .arg(dst_active ? "Y" : "N");
  }

  const bool segment_active = active_connected > threshold;
  QString out;
  out += QString("Src column (%1,%2)  cell=%3  seg=%4\n").arg(src_col_x).arg(src_col_y).arg(src_cell).arg(src_segment);
  out += QString("Cell state (now): active=%1  predictive=%2  learning=%3\n")
             .arg(cell_active ? "Y" : "N")
             .arg(cell_predictive ? "Y" : "N")
             .arg(cell_learning ? "Y" : "N");
  out += QString("Selected segment active: %1  (active_connected=%2  threshold=%3)\n\n")
             .arg(segment_active ? "Y" : "N")
             .arg(active_connected)
             .arg(threshold);
  out += QString("Synapses: %1  connected: %2  tgt_active: %3\n\n")
             .arg(int(syns.size()))
             .arg(connected_count)
             .arg(tgt_active_total);
  out += syn_lines;
  return out;
}

void MainWindow::updatePinnedDistalPanels() {
  auto it = pinned_distal_views_.begin();
  while (it != pinned_distal_views_.end()) {
    if (it->dock.isNull() || it->text.isNull()) {
      it = pinned_distal_views_.erase(it);
      continue;
    }

    const int grid_w = snapshot_.columns_shape.cols;
    const int grid_h = snapshot_.columns_shape.rows;
    const bool src_col_valid = (it->col_x >= 0 && it->col_x < grid_w && it->col_y >= 0 && it->col_y < grid_h);
    const bool src_cell_valid = (it->cell >= 0 && it->cell < snapshot_.cells_per_column);

    if (!src_col_valid || !src_cell_valid) {
      it->text->setPlainText("Pinned source is no longer valid for this snapshot.");
      ++it;
      continue;
    }

    const int nsegs = runtime_.num_segments(it->col_x, it->col_y, it->cell);
    if (it->segment < 0 || it->segment >= nsegs) {
      it->text->setPlainText("Pinned segment is no longer available.");
      ++it;
      continue;
    }

    const auto query = runtime_.query_distal(it->col_x, it->col_y, it->cell, it->segment);
    it->text->setPlainText(formatDistalSynapseText(query, it->col_x, it->col_y, it->cell, it->segment));
    ++it;
  }
}

void MainWindow::updatePinnedProximalPanels() {
  auto it = pinned_proximal_views_.begin();
  while (it != pinned_proximal_views_.end()) {
    if (it->dock.isNull() || it->text.isNull()) {
      it = pinned_proximal_views_.erase(it);
      continue;
    }

    const int grid_w = snapshot_.columns_shape.cols;
    const int grid_h = snapshot_.columns_shape.rows;
    const bool src_col_valid = (it->col_x >= 0 && it->col_x < grid_w && it->col_y >= 0 && it->col_y < grid_h);
    if (!src_col_valid) {
      it->text->setPlainText("Pinned column is no longer valid for this snapshot.");
      ++it;
      continue;
    }

    const auto query = runtime_.query_proximal(it->col_x, it->col_y);
    it->text->setPlainText(formatProximalSynapseText(query, it->col_x, it->col_y));
    ++it;
  }
}

void MainWindow::pinCurrentProximal() {
  if (selected_col_x_ < 0 || selected_col_y_ < 0) {
    statusBar()->showMessage("Select a column before pinning proximal synapses.", 3000);
    return;
  }

  const auto query = runtime_.query_proximal(selected_col_x_, selected_col_y_);
  auto* text = new QPlainTextEdit(this);
  text->setReadOnly(true);
  text->setPlainText(formatProximalSynapseText(query, selected_col_x_, selected_col_y_));

  auto* dock =
      new QDockWidget(QString("Proximal synapses (%1,%2)").arg(selected_col_x_).arg(selected_col_y_), this);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
  dock->setWidget(text);
  dock->setAttribute(Qt::WA_DeleteOnClose, true);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  dock->show();

  pinned_proximal_views_.push_back(PinnedProximalView{selected_col_x_, selected_col_y_, text, dock});
}

void MainWindow::pinCurrentDistal() {
  if (selected_col_x_ < 0 || selected_col_y_ < 0 || selected_cell_ < 0 || selected_segment_ < 0) {
    statusBar()->showMessage("Select a column, cell, and segment before pinning distal synapses.", 3000);
    return;
  }

  const int nsegs = runtime_.num_segments(selected_col_x_, selected_col_y_, selected_cell_);
  if (selected_segment_ >= nsegs) {
    statusBar()->showMessage("Current segment is no longer valid.", 3000);
    return;
  }

  const auto query = runtime_.query_distal(selected_col_x_, selected_col_y_, selected_cell_, selected_segment_);
  auto* text = new QPlainTextEdit(this);
  text->setReadOnly(true);
  text->setPlainText(formatDistalSynapseText(query, selected_col_x_, selected_col_y_, selected_cell_, selected_segment_));

  auto* dock = new QDockWidget(
      QString("Distal synapses (%1,%2 c%3 s%4)").arg(selected_col_x_).arg(selected_col_y_).arg(selected_cell_).arg(selected_segment_),
      this);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
  dock->setWidget(text);
  dock->setAttribute(Qt::WA_DeleteOnClose, true);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  dock->show();

  pinned_distal_views_.push_back(
      PinnedDistalView{selected_col_x_, selected_col_y_, selected_cell_, selected_segment_, text, dock});
}

QString MainWindow::layoutFilePath() {
  const QString config_dir =
      QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation) + "/htm_gui";
  return config_dir + "/layout.dat";
}

void MainWindow::saveLayout() {
  const QString path = layoutFilePath();
  QDir().mkpath(QFileInfo(path).absolutePath());

  QFile file(path);
  if (!file.open(QIODevice::WriteOnly)) {
    statusBar()->showMessage(QString("Failed to save layout: %1").arg(file.errorString()), 5000);
    return;
  }

  QDataStream out(&file);
  out << quint32(0x48544D4C);  // "HTMLAYOUT" magic
  out << quint16(1);           // version
  out << saveGeometry();
  out << saveState();

  statusBar()->showMessage(QString("Layout saved to %1").arg(path), 3000);
}

void MainWindow::restoreLayout() {
  const QString path = layoutFilePath();
  QFile file(path);
  if (!file.exists()) {
    statusBar()->showMessage("No saved layout found.", 3000);
    return;
  }
  if (!file.open(QIODevice::ReadOnly)) {
    statusBar()->showMessage(QString("Failed to read layout: %1").arg(file.errorString()), 5000);
    return;
  }

  QDataStream in(&file);
  quint32 magic{};
  quint16 version{};
  in >> magic >> version;

  if (magic != 0x48544D4C || version != 1) {
    statusBar()->showMessage("Saved layout file is corrupt or incompatible; ignoring.", 5000);
    return;
  }

  QByteArray geometry;
  QByteArray state;
  in >> geometry >> state;

  restoreGeometry(geometry);
  if (!restoreState(state)) {
    statusBar()->showMessage("Could not fully restore layout (dock mismatch); using defaults.", 5000);
    return;
  }
  statusBar()->showMessage("Layout restored.", 3000);
}

void MainWindow::markState() {
  // Deep-copy the input SDR so the marked window is truly frozen.
  htm_gui::Snapshot frozen = snapshot_;
  if (snapshot_.input) {
    frozen.input = std::make_shared<const std::vector<int>>(*snapshot_.input);
  }

  auto* w = new FrozenWindow(std::move(frozen), nullptr);
  w->setAttribute(Qt::WA_DeleteOnClose, true);
  w->show();
  marked_windows_.push_back(w);
}

QImage MainWindow::renderInput(const htm_gui::Snapshot& s, int max_w, int max_h) const {
  const int src_w = s.input_shape.cols;
  const int src_h = s.input_shape.rows;

  const int out_w = std::max(1, std::min(max_w, src_w));
  const int out_h = std::max(1, std::min(max_h, src_h));

  QImage img(out_w, out_h, QImage::Format_ARGB32);

  if (!s.input) {
    img.fill(red());
    return img;
  }

  for (int oy = 0; oy < out_h; ++oy) {
    const int sy = (oy * src_h) / out_h;
    for (int ox = 0; ox < out_w; ++ox) {
      const int sx = (ox * src_w) / out_w;
      const int v = (*s.input)[sy * src_w + sx];
      img.setPixelColor(ox, oy, v ? green() : red());
    }
  }

  // Overlay potential/connected proximal synapses for selected column.
  if (proximal_query_.has_value() && selected_col_x_ >= 0 && selected_col_y_ >= 0) {
    for (const auto& syn : proximal_query_->synapses) {
      if (syn.input_x < 0 || syn.input_x >= src_w || syn.input_y < 0 || syn.input_y >= src_h) {
        continue;
      }

      const int ox = (syn.input_x * out_w) / src_w;
      const int oy = (syn.input_y * out_h) / src_h;

      if (ox < 0 || ox >= out_w || oy < 0 || oy >= out_h) {
        continue;
      }

      if (syn.connected) {
        img.setPixelColor(ox, oy, syn.input_value ? dark_green() : transp_blue());
      } else {
        img.setPixelColor(ox, oy, syn.input_value ? light_green() : light_red());
      }
    }
  }

  return img;
}

QImage MainWindow::renderColumns(const htm_gui::Snapshot& s, int max_w, int max_h) const {
  return renderColumnsWithCellOverlay(
      s, max_w, max_h, selected_col_x_, selected_col_y_, &distal_overlay_, show_cell_overlay_);
}

QImage MainWindow::renderCells(const htm_gui::Snapshot& s, int col_x, int col_y, int max_size) const {
  const int cells = s.cells_per_column;
  if (cells <= 0 || cells > 64) {
    QImage img(1, 1, QImage::Format_ARGB32);
    img.fill(black());
    return img;
  }

  const int col_idx = htm_gui::flatten_xy(col_x, col_y, s.columns_shape.cols);
  if (col_idx < 0 || col_idx >= int(s.column_cell_masks.size())) {
    QImage img(1, 1, QImage::Format_ARGB32);
    img.fill(black());
    return img;
  }

  const auto masks = s.column_cell_masks[col_idx];

  const int side = int(std::ceil(std::sqrt(double(cells))));
  const int out = std::max(1, std::min(max_size, side * 12));
  QImage img(out, out, QImage::Format_ARGB32);
  img.fill(QColor(0, 0, 0, 0));

  const int cell_px = std::max(1, out / side);

  for (int c = 0; c < cells; ++c) {
    const int gx = c % side;
    const int gy = c / side;

    QColor color = QColor(0, 0, 0, 0x20);
    switch (cell_mode_) {
      case CellDisplayMode::Active:
        color = (masks.active & (std::uint64_t(1) << c)) ? blue() : transp_blue();
        break;
      case CellDisplayMode::Predictive:
        color = (masks.predictive & (std::uint64_t(1) << c)) ? black() : transp_blue();
        break;
      case CellDisplayMode::Learning:
        color = (masks.learning & (std::uint64_t(1) << c)) ? dark_green() : transp_blue();
        break;
    }

    for (int py = gy * cell_px; py < std::min(out, (gy + 1) * cell_px); ++py) {
      for (int px = gx * cell_px; px < std::min(out, (gx + 1) * cell_px); ++px) {
        img.setPixelColor(px, py, color);
      }
    }
  }

  // Highlight selected cell (white square).
  if (selected_cell_ >= 0 && selected_cell_ < cells) {
    const int gx = selected_cell_ % side;
    const int gy = selected_cell_ / side;
    const int x0 = gx * cell_px;
    const int y0 = gy * cell_px;
    const int x1 = std::min(out, (gx + 1) * cell_px);
    const int y1 = std::min(out, (gy + 1) * cell_px);
    if (x1 - x0 <= 1 || y1 - y0 <= 1) {
      img.setPixelColor(x0, y0, QColor(0xFF, 0xFF, 0xFF));
    } else {
      for (int py = y0; py < y1; ++py) {
        for (int px = x0; px < x1; ++px) {
          const bool border = (px == x0 || px == x1 - 1 || py == y0 || py == y1 - 1);
          if (border) {
            img.setPixelColor(px, py, QColor(0xFF, 0xFF, 0xFF));
          }
        }
      }
    }
  }

  return img;
}

}  // namespace htm_gui::qt
