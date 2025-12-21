#pragma once

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QObject>
#include <QResizeEvent>

namespace htm_gui::qt {

class ImageView final : public QGraphicsView {
  Q_OBJECT

public:
  explicit ImageView(QWidget* parent = nullptr);

  void setImage(const QImage& image);

  // Provide a mapping from displayed image pixels to an underlying logical grid.
  void setLogicalGrid(int cols, int rows);

signals:
  void clicked(int logical_x, int logical_y);

protected:
  void mousePressEvent(QMouseEvent* event) override;
  void resizeEvent(QResizeEvent* event) override;

private:
  void refit();

  QGraphicsScene scene_;
  QGraphicsPixmapItem* pixmap_{nullptr};
  QImage image_;
  int logical_cols_{0};
  int logical_rows_{0};
};

}  // namespace htm_gui::qt
