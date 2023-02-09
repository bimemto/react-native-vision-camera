package com.mrousavy.camera.helpers

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark
import com.google.mlkit.vision.face.FaceLandmark.LandmarkType
import java.util.*

class FaceGraphic internal constructor(
  overlay: GraphicOverlay?,
  private val face: Face,
  private val name: String
) :
  GraphicOverlay.Graphic(overlay!!) {
  private val facePositionPaint: Paint
  private val idPaints: Array<Paint?>
  private val boxPaints: Array<Paint?>
  private val labelPaints: Array<Paint?>

  init {
    val selectedColor = Color.WHITE
    facePositionPaint = Paint()
    facePositionPaint.color = selectedColor
    val numColors = COLORS.size
    idPaints = arrayOfNulls(numColors)
    boxPaints = arrayOfNulls(numColors)
    labelPaints = arrayOfNulls(numColors)
    for (i in 0 until numColors) {
      idPaints[i] = Paint()
      idPaints[i]!!.color = COLORS[i][0]
      idPaints[i]!!.textSize = ID_TEXT_SIZE
      boxPaints[i] = Paint()
      boxPaints[i]!!.color = COLORS[i][1]
      boxPaints[i]!!.style = Paint.Style.STROKE
      boxPaints[i]!!.strokeWidth = BOX_STROKE_WIDTH
      labelPaints[i] = Paint()
      labelPaints[i]!!.color = COLORS[i][1]
      labelPaints[i]!!.style = Paint.Style.FILL
    }
  }

  /** Draws the face annotations for position on the supplied canvas.  */
  override fun draw(canvas: Canvas?) {
    val face = face ?: return

    // Draws a circle at the position of the detected face, with the face's track id below.
    val x: Float = translateX(face.boundingBox.centerX().toFloat())
    val y: Float = translateY(face.boundingBox.centerY().toFloat())

    // Calculate positions.
    val left: Float = x - scale(face.boundingBox.width() / 2.0f)
    val top: Float = y - scale(face.boundingBox.height() / 2.0f)
    val right: Float = x + scale(face.boundingBox.width() / 2.0f)
    val bottom: Float = y + scale(face.boundingBox.height() / 2.0f)
    val lineHeight = ID_TEXT_SIZE + BOX_STROKE_WIDTH
    var yLabelOffset: Float = if (face.trackingId == null) 0f else -lineHeight
    var colorID = 9
    if (name === "Stranger" || name === "Face spoofing detected" || name === "Please look straight at the camera") {
      colorID = 3
    }

    // Calculate width and height of label box
    var textWidth = idPaints[colorID]!!.measureText("ID: " + face.trackingId)
    if (face.smilingProbability != null) {
      yLabelOffset -= lineHeight
      textWidth = Math.max(
        textWidth,
        idPaints[colorID]!!
          .measureText(String.format(Locale.US, "Happiness: %.2f", face.smilingProbability))
      )
    }
    if (face.leftEyeOpenProbability != null) {
      yLabelOffset -= lineHeight
      textWidth = Math.max(
        textWidth,
        idPaints[colorID]!!.measureText(
          String.format(
            Locale.US, "Left eye open: %.2f", face.leftEyeOpenProbability
          )
        )
      )
    }
    if (face.rightEyeOpenProbability != null) {
      yLabelOffset -= lineHeight
      textWidth = Math.max(
        textWidth,
        idPaints[colorID]!!.measureText(
          String.format(
            Locale.US, "Right eye open: %.2f", face.rightEyeOpenProbability
          )
        )
      )
    }
    yLabelOffset = yLabelOffset - 1 * lineHeight
    textWidth = Math.max(textWidth, idPaints[colorID]!!.measureText(name))

    /*yLabelOffset = yLabelOffset - 3 * lineHeight;
    textWidth =
            Math.max(
                    textWidth,
                    idPaints[colorID].measureText(
                            String.format(Locale.US, "EulerX: %.2f", face.getHeadEulerAngleX())));
    textWidth =
            Math.max(
                    textWidth,
                    idPaints[colorID].measureText(
                            String.format(Locale.US, "EulerY: %.2f", face.getHeadEulerAngleY())));
    textWidth =
            Math.max(
                    textWidth,
                    idPaints[colorID].measureText(
                            String.format(Locale.US, "EulerZ: %.2f", face.getHeadEulerAngleZ())));

    canvas.drawText(
            "EulerX: " + face.getHeadEulerAngleX(), left, top + yLabelOffset, idPaints[colorID]);
    yLabelOffset += lineHeight;
    canvas.drawText(
            "EulerY: " + face.getHeadEulerAngleY(), left, top + yLabelOffset, idPaints[colorID]);
    yLabelOffset += lineHeight;
    canvas.drawText(
            "EulerZ: " + face.getHeadEulerAngleZ(), left, top + yLabelOffset, idPaints[colorID]);*/

    // Draw labels
    canvas?.drawRect(
      left - BOX_STROKE_WIDTH,
      top + yLabelOffset,
      left + textWidth + 2 * BOX_STROKE_WIDTH,
      top,
      labelPaints[colorID]!!
    )
    yLabelOffset += ID_TEXT_SIZE
    canvas?.drawRect(left, top, right, bottom, boxPaints[colorID]!!)
    if (face.trackingId != null) {
      canvas?.drawText(
        "ID: " + face.trackingId, left, top + yLabelOffset,
        idPaints[colorID]!!
      )
      yLabelOffset += lineHeight
    }

    // Draws all face contours.
    for (contour in face.allContours) {
      for (point in contour.points) {
        canvas?.drawCircle(
          translateX(point.x), translateY(point.y), FACE_POSITION_RADIUS, facePositionPaint
        )
      }
    }

    // Draws smiling and left/right eye open probabilities.
    if (face.smilingProbability != null) {
      canvas?.drawText(
        "Smiling: " + String.format(Locale.US, "%.2f", face.smilingProbability),
        left,
        top + yLabelOffset,
        idPaints[colorID]!!
      )
      yLabelOffset += lineHeight
    }
    val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
    if (face.leftEyeOpenProbability != null) {
      canvas?.drawText(
        "Left eye open: " + String.format(Locale.US, "%.2f", face.leftEyeOpenProbability),
        left,
        top + yLabelOffset,
        idPaints[colorID]!!
      )
      yLabelOffset += lineHeight
    }
    if (leftEye != null) {
      val leftEyeLeft: Float = translateX(leftEye.position.x) - idPaints[colorID]!!
        .measureText("Left Eye") / 2.0f
      canvas?.drawRect(
        leftEyeLeft - BOX_STROKE_WIDTH,
        translateY(leftEye.position.y) + ID_Y_OFFSET - ID_TEXT_SIZE,
        leftEyeLeft + idPaints[colorID]!!.measureText("Left Eye") + BOX_STROKE_WIDTH,
        translateY(leftEye.position.y) + ID_Y_OFFSET + BOX_STROKE_WIDTH,
        labelPaints[colorID]!!
      )
      canvas?.drawText(
        "Left Eye",
        leftEyeLeft,
        translateY(leftEye.position.y) + ID_Y_OFFSET,
        idPaints[colorID]!!
      )
    }
    val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
    if (face.rightEyeOpenProbability != null) {
      canvas?.drawText(
        "Right eye open: " + String.format(Locale.US, "%.2f", face.rightEyeOpenProbability),
        left,
        top + yLabelOffset,
        idPaints[colorID]!!
      )
      yLabelOffset += lineHeight
    }
    if (rightEye != null) {
      val rightEyeLeft: Float = translateX(rightEye.position.x) - idPaints[colorID]!!
        .measureText("Right Eye") / 2.0f
      canvas?.drawRect(
        rightEyeLeft - BOX_STROKE_WIDTH,
        translateY(rightEye.position.y) + ID_Y_OFFSET - ID_TEXT_SIZE,
        rightEyeLeft + idPaints[colorID]!!.measureText("Right Eye") + BOX_STROKE_WIDTH,
        translateY(rightEye.position.y) + ID_Y_OFFSET + BOX_STROKE_WIDTH,
        labelPaints[colorID]!!
      )
      canvas?.drawText(
        "Right Eye",
        rightEyeLeft,
        translateY(rightEye.position.y) + ID_Y_OFFSET,
        idPaints[colorID]!!
      )
    }
    canvas?.drawText(name, left, top + yLabelOffset, idPaints[colorID]!!)

    // Draw facial landmarks
    drawFaceLandmark(canvas!!, FaceLandmark.LEFT_EYE)
    drawFaceLandmark(canvas, FaceLandmark.RIGHT_EYE)
    drawFaceLandmark(canvas, FaceLandmark.LEFT_CHEEK)
    drawFaceLandmark(canvas, FaceLandmark.RIGHT_CHEEK)
  }

  private fun drawFaceLandmark(canvas: Canvas, @LandmarkType landmarkType: Int) {
    val faceLandmark = face.getLandmark(landmarkType)
    if (faceLandmark != null) {
      canvas.drawCircle(
        translateX(faceLandmark.position.x),
        translateY(faceLandmark.position.y),
        FACE_POSITION_RADIUS,
        facePositionPaint
      )
    }
  }

  companion object {
    private const val FACE_POSITION_RADIUS = 8.0f
    private const val ID_TEXT_SIZE = 30.0f
    private const val ID_Y_OFFSET = 40.0f
    private const val BOX_STROKE_WIDTH = 5.0f
    private val COLORS = arrayOf(
      intArrayOf(Color.BLACK, Color.WHITE), intArrayOf(Color.WHITE, Color.MAGENTA), intArrayOf(
        Color.BLACK, Color.LTGRAY
      ), intArrayOf(Color.WHITE, Color.RED), intArrayOf(Color.WHITE, Color.BLUE), intArrayOf(
        Color.WHITE, Color.DKGRAY
      ), intArrayOf(Color.BLACK, Color.CYAN), intArrayOf(Color.BLACK, Color.YELLOW), intArrayOf(
        Color.WHITE, Color.BLACK
      ), intArrayOf(Color.BLACK, Color.GREEN)
    )
  }

}
