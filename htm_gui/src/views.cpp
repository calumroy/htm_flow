#include "views.hpp"

#include <algorithm>

#include <QApplication>
#include <QMouseEvent>
#include <QPixmap>

namespace htm_gui::qt {

ImageView::ImageView(QWidget* parent) : QGraphicsView(parent) {
  setScene(&scene_);
  setDragMode(QGraphicsView::ScrollHandDrag);
  setRenderHint(QPainter::Antialiasing, false);
  setRenderHint(QPainter::SmoothPixmapTransform, false);
  setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
  setResizeAnchor(QGraphicsView::AnchorUnderMouse);
  setFocusPolicy(Qt::StrongFocus);
  pixmap_ = scene_.addPixmap(QPixmap());
}

void ImageView::setLogicalGrid(int cols, int rows) {
  logical_cols_ = cols;
  logical_rows_ = rows;
}

void ImageView::setImage(const QImage& image) {
  const bool size_changed = (!has_image_ || image_.size() != image.size());
  image_ = image;
  has_image_ = !image_.isNull();
  pixmap_->setPixmap(QPixmap::fromImage(image_));
  scene_.setSceneRect(QRectF(QPointF(0, 0), QSizeF(image_.width(), image_.height())));

  // Only auto-refit when we haven't manually zoomed/panned and the underlying image size changes.
  if (auto_fit_ && size_changed) {
    refit();
  }
}

void ImageView::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    left_pressed_ = true;
    left_press_pos_ = event->pos();
    setFocus(Qt::MouseFocusReason);
  }

  QGraphicsView::mousePressEvent(event);
}

void ImageView::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton && left_pressed_ && logical_cols_ > 0 && logical_rows_ > 0 && !image_.isNull()) {
    left_pressed_ = false;

    // If the mouse moved far enough to be considered a drag, don't treat it as a click.
    const int dist = (event->pos() - left_press_pos_).manhattanLength();
    if (dist < QApplication::startDragDistance()) {
      const QPointF scene_pt = mapToScene(event->pos());

      const double ix = std::clamp(scene_pt.x(), 0.0, double(image_.width() - 1));
      const double iy = std::clamp(scene_pt.y(), 0.0, double(image_.height() - 1));

      const int logical_x = std::clamp(int(ix * logical_cols_ / double(image_.width())), 0, logical_cols_ - 1);
      const int logical_y = std::clamp(int(iy * logical_rows_ / double(image_.height())), 0, logical_rows_ - 1);

      emit clicked(logical_x, logical_y);
    }
  }

  QGraphicsView::mouseReleaseEvent(event);
}

void ImageView::wheelEvent(QWheelEvent* event) {
  // Ctrl+mouse-wheel zoom (bonus). Regular wheel keeps its normal scroll behavior.
  if (event->modifiers().testFlag(Qt::ControlModifier)) {
    const QPoint num_degrees = event->angleDelta() / 8;
    if (!num_degrees.isNull()) {
      const int dy = num_degrees.y();
      if (dy > 0) {
        zoomIn();
      } else if (dy < 0) {
        zoomOut();
      }
    }
    event->accept();
    return;
  }

  QGraphicsView::wheelEvent(event);
}

void ImageView::resizeEvent(QResizeEvent* event) {
  QGraphicsView::resizeEvent(event);
  if (auto_fit_) {
    refit();
  }
}

void ImageView::refit() {
  if (!image_.isNull()) {
    // Reset to a clean baseline so zoom clamping stays meaningful.
    resetTransform();
    fitInView(scene_.sceneRect(), Qt::KeepAspectRatio);
    zoom_ = 1.0;
  }
}

void ImageView::markManualViewChange() {
  auto_fit_ = false;
}

void ImageView::applyZoomFactor(double factor) {
  if (image_.isNull() || factor <= 0.0) {
    return;
  }

  markManualViewChange();

  constexpr double kMinZoom = 0.1;
  constexpr double kMaxZoom = 40.0;

  double new_zoom = zoom_ * factor;
  if (new_zoom < kMinZoom) {
    factor = kMinZoom / zoom_;
    new_zoom = kMinZoom;
  } else if (new_zoom > kMaxZoom) {
    factor = kMaxZoom / zoom_;
    new_zoom = kMaxZoom;
  }

  zoom_ = new_zoom;
  scale(factor, factor);
}

void ImageView::zoomIn() {
  applyZoomFactor(1.25);
}

void ImageView::zoomOut() {
  applyZoomFactor(1.0 / 1.25);
}

void ImageView::resetZoom() {
  auto_fit_ = true;
  refit();
}

}  // namespace htm_gui::qt
