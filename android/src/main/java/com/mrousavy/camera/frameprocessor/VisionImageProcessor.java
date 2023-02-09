package com.mrousavy.camera.frameprocessor;

import androidx.camera.core.ImageProxy;

import com.google.mlkit.common.MlKitException;
import com.mrousavy.camera.helpers.FrameMetadata;
import com.mrousavy.camera.helpers.GraphicOverlay;

import java.nio.ByteBuffer;

public interface VisionImageProcessor {

    /** Processes ByteBuffer image data, e.g. used for Camera1 live preview case. */
    void processByteBuffer(
            ByteBuffer data, FrameMetadata frameMetadata, GraphicOverlay graphicOverlay)
            throws MlKitException;

    /** Processes ImageProxy image data, e.g. used for CameraX live preview case. */
    void processImageProxy(ImageProxy image, GraphicOverlay graphicOverlay) throws MlKitException;

    /** Stops the underlying machine learning model and release resources. */
    void stop();
}
