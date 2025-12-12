#include "views.hpp"

#include <algorithm>

#include <QMouseEvent>
#include <QPixmap>

namespace htm_gui::qt {

ImageView::ImageView(QWidget* parent) : QGraphicsView(parent) {
  setScene(&scene_);
  setDragMode(QGraphicsView::ScrollHandDrag);
  setRenderHint(QPainter::Antialiasing, false);
  setRenderHint(QPainter::SmoothPixmapTransform, false);
  pixmap_ = scene_.addPixmap(QPixmap());
}

void ImageView::setLogicalGrid(int cols, int rows) {
  logical_cols_ = cols;
  logical_rows_ = rows;
}

void ImageView::setImage(const QImage& image) {
  image_ = image;
  pixmap_->setPixmap(QPixmap::fromImage(image_));
  scene_.setSceneRect(QRectF(QPointF(0, 0), QSizeF(image_.width(), image_.height())));
  refit();
}

void ImageView::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton && logical_cols_ > 0 && logical_rows_ > 0 && !image_.isNull()) {
    const QPointF scene_pt = mapToScene(event->pos());

    const double ix = std::clamp(scene_pt.x(), 0.0, double(image_.width() - 1));
    const double iy = std::clamp(scene_pt.y(), 0.0, double(image_.height() - 1));

    const int logical_x = std::clamp(int(ix * logical_cols_ / double(image_.width())), 0, logical_cols_ - 1);
    const int logical_y = std::clamp(int(iy * logical_rows_ / double(image_.height())), 0, logical_rows_ - 1);

    emit clicked(logical_x, logical_y);
  }

  QGraphicsView::mousePressEvent(event);
}

void ImageView::resizeEvent(QResizeEvent* event) {
  QGraphicsView::resizeEvent(event);
  refit();
}

void ImageView::refit() {
  if (!image_.isNull()) {
    fitInView(scene_.sceneRect(), Qt::KeepAspectRatio);
  }
}

}  // namespace htm_gui::qt
