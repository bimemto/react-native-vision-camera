package com.mrousavy.camera.helpers

import androidx.camera.core.ImageProxy
import com.google.mlkit.common.MlKitException
import java.nio.ByteBuffer

interface VisionImageProcessor {
  /** Processes ByteBuffer image data, e.g. used for Camera1 live preview case.  */
  @Throws(MlKitException::class)
  fun processByteBuffer(
    data: ByteBuffer?, frameMetadata: FrameMetadata?, graphicOverlay: GraphicOverlay?
  )

  /** Processes ImageProxy image data, e.g. used for CameraX live preview case.  */
  @Throws(MlKitException::class)
  fun processImageProxy(image: ImageProxy?, graphicOverlay: GraphicOverlay?)

  /** Stops the underlying machine learning model and release resources.  */
  fun stop()
}
