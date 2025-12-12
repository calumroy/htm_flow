#include "main_window.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include <QAction>
#include <QHBoxLayout>
#include <QImage>
#include <QInputDialog>
#include <QStatusBar>
#include <QToolBar>
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

class FrozenWindow final : public QMainWindow {
public:
  explicit FrozenWindow(htm_gui::Snapshot snapshot, QWidget* parent = nullptr) : QMainWindow(parent), snapshot_(std::move(snapshot)) {
    setWindowTitle(QString("Marked state (t=%1)").arg(snapshot_.timestep));

    auto* central = new QWidget(this);
    auto* layout = new QHBoxLayout(central);

    input_view_ = new htm_gui::qt::ImageView(this);
    columns_view_ = new htm_gui::qt::ImageView(this);
    cells_view_ = new htm_gui::qt::ImageView(this);

    layout->addWidget(input_view_, 1);
    layout->addWidget(columns_view_, 1);
    layout->addWidget(cells_view_, 0);

    setCentralWidget(central);
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
    const int src_w = s.columns_shape.cols;
    const int src_h = s.columns_shape.rows;

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

MainWindow::MainWindow(htm_gui::IHtmRuntime& runtime, QWidget* parent)
    : QMainWindow(parent), runtime_(runtime) {
  setWindowTitle(QString::fromStdString(runtime_.name()));

  auto* central = new QWidget(this);
  auto* layout = new QHBoxLayout(central);

  input_view_ = new ImageView(this);
  columns_view_ = new ImageView(this);
  cells_view_ = new ImageView(this);

  layout->addWidget(input_view_, 1);
  layout->addWidget(columns_view_, 1);
  layout->addWidget(cells_view_, 0);

  setCentralWidget(central);

  auto* tb = addToolBar("Controls");
  auto* step_one = new QAction("Step", this);
  auto* step_n = new QAction("N steps", this);
  auto* show_active = new QAction("Active Cells", this);
  auto* show_pred = new QAction("Predict Cells", this);
  auto* show_learn = new QAction("Learn Cells", this);
  auto* mark = new QAction("Mark", this);
  tb->addAction(step_one);
  tb->addAction(step_n);
  tb->addSeparator();
  tb->addAction(show_active);
  tb->addAction(show_pred);
  tb->addAction(show_learn);
  tb->addSeparator();
  tb->addAction(mark);

  connect(step_one, &QAction::triggered, this, &MainWindow::stepOne);
  connect(step_n, &QAction::triggered, this, &MainWindow::stepN);
  connect(show_active, &QAction::triggered, this, &MainWindow::showActiveCells);
  connect(show_pred, &QAction::triggered, this, &MainWindow::showPredictCells);
  connect(show_learn, &QAction::triggered, this, &MainWindow::showLearnCells);
  connect(mark, &QAction::triggered, this, &MainWindow::markState);

  connect(columns_view_, &ImageView::clicked, this, &MainWindow::onColumnClicked);
  connect(cells_view_, &ImageView::clicked, this, &MainWindow::onCellClicked);

  statusBar()->showMessage("Ready");
  refresh();
}

void MainWindow::refresh() {
  snapshot_ = runtime_.snapshot();

  if (snapshot_.input_shape.rows <= 0 || snapshot_.input_shape.cols <= 0 || snapshot_.columns_shape.rows <= 0 ||
      snapshot_.columns_shape.cols <= 0) {
    statusBar()->showMessage("Invalid snapshot shapes");
    return;
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

  const std::string msg = "t=" + std::to_string(snapshot_.timestep) +
                          " active_cols=" + std::to_string(snapshot_.active_column_indices.size()) +
                          " grid=" + std::to_string(snapshot_.columns_shape.cols) + "x" +
                          std::to_string(snapshot_.columns_shape.rows);
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
  if (selected_col_x_ >= 0 && selected_col_y_ >= 0) {
    const auto q = runtime_.query_proximal(selected_col_x_, selected_col_y_);
    for (const auto& syn : q.synapses) {
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
  const int src_w = s.columns_shape.cols;
  const int src_h = s.columns_shape.rows;

  const int out_w = std::max(1, std::min(max_w, src_w));
  const int out_h = std::max(1, std::min(max_h, src_h));

  std::unordered_set<int> active;
  active.reserve(s.active_column_indices.size());
  for (int idx : s.active_column_indices) {
    active.insert(idx);
  }

  QImage img(out_w, out_h, QImage::Format_ARGB32);
  for (int oy = 0; oy < out_h; ++oy) {
    const int sy = (oy * src_h) / out_h;
    for (int ox = 0; ox < out_w; ++ox) {
      const int sx = (ox * src_w) / out_w;
      const int idx = htm_gui::flatten_xy(sx, sy, src_w);
      img.setPixelColor(ox, oy, active.count(idx) ? green() : red());
    }
  }

  // Overlay distal synapse targets (connected) if a segment is selected.
  if (distal_overlay_.has_value()) {
    for (const auto& syn : distal_overlay_->synapses) {
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

  // Highlight selected column if it maps to a pixel in the downsample.
  if (selected_col_x_ >= 0 && selected_col_y_ >= 0) {
    const int px = (selected_col_x_ * out_w) / src_w;
    const int py = (selected_col_y_ * out_h) / src_h;
    for (int dx = -2; dx <= 2; ++dx) {
      const int x = px + dx;
      if (x >= 0 && x < out_w && py >= 0 && py < out_h) {
        img.setPixelColor(x, py, yellow());
      }
    }
    for (int dy = -2; dy <= 2; ++dy) {
      const int y = py + dy;
      if (px >= 0 && px < out_w && y >= 0 && y < out_h) {
        img.setPixelColor(px, y, yellow());
      }
    }
  }

  return img;
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
    for (int py = gy * cell_px; py < std::min(out, (gy + 1) * cell_px); ++py) {
      for (int px = gx * cell_px; px < std::min(out, (gx + 1) * cell_px); ++px) {
        img.setPixelColor(px, py, QColor(0xFF, 0xFF, 0xFF));
      }
    }
  }

  return img;
}

}  // namespace htm_gui::qt
