#pragma once

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QObject>
#include <QResizeEvent>
#include <QWheelEvent>

namespace htm_gui::qt {

class ImageView final : public QGraphicsView {
  Q_OBJECT

public:
  explicit ImageView(QWidget* parent = nullptr);

  void setImage(const QImage& image);

  // Provide a mapping from displayed image pixels to an underlying logical grid.
  void setLogicalGrid(int cols, int rows);

  void zoomIn();
  void zoomOut();
  void resetZoom();

signals:
  void clicked(int logical_x, int logical_y);

protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void resizeEvent(QResizeEvent* event) override;

private:
  void refit();
  void applyZoomFactor(double factor);
  void markManualViewChange();

  QGraphicsScene scene_;
  QGraphicsPixmapItem* pixmap_{nullptr};
  QImage image_;
  int logical_cols_{0};
  int logical_rows_{0};

  bool has_image_{false};
  bool auto_fit_{true};
  double zoom_{1.0};

  bool left_pressed_{false};
  QPoint left_press_pos_;
};

}  // namespace htm_gui::qt
