package com.mrousavy.camera

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel
import android.Manifest
import android.annotation.SuppressLint
import android.app.Dialog
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.*
import android.graphics.drawable.ColorDrawable
import android.hardware.camera2.*
import android.media.Image
import android.util.Log
import android.util.Range
import android.view.*
import android.view.View.OnTouchListener
import android.widget.*
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.core.impl.*
import androidx.camera.extensions.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.constraintlayout.widget.ConstraintSet
import androidx.core.content.ContextCompat
import androidx.lifecycle.*
import com.facebook.jni.HybridData
import com.facebook.proguard.annotations.DoNotStrip
import com.facebook.react.bridge.*
import com.google.mlkit.common.MlKitException
import com.mrousavy.camera.frameprocessor.FaceDetectorProcessor
import com.mrousavy.camera.frameprocessor.FrameProcessorPerformanceDataCollector
import com.mrousavy.camera.frameprocessor.FrameProcessorRuntimeManager
import com.mrousavy.camera.helpers.GraphicOverlay
import com.mrousavy.camera.models.Person
import com.mrousavy.camera.utils.*
import kotlinx.coroutines.*
import kotlinx.coroutines.guava.await
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ReadOnlyBufferException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.experimental.inv
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min


//
// TODOs for the CameraView which are currently too hard to implement either because of CameraX' limitations, or my brain capacity.
//
// CameraView
// TODO: Actually use correct sizes for video and photo (currently it's both the video size)
// TODO: Configurable FPS higher than 30
// TODO: High-speed video recordings (export in CameraViewModule::getAvailableVideoDevices(), and set in CameraView::configurePreview()) (120FPS+)
// TODO: configureSession() enableDepthData
// TODO: configureSession() enableHighQualityPhotos
// TODO: configureSession() enablePortraitEffectsMatteDelivery
// TODO: configureSession() colorSpace

// CameraView+RecordVideo
// TODO: Better startRecording()/stopRecording() (promise + callback, wait for TurboModules/JSI)
// TODO: videoStabilizationMode
// TODO: Return Video size/duration

// CameraView+TakePhoto
// TODO: Mirror selfie images
// TODO: takePhoto() depth data
// TODO: takePhoto() raw capture
// TODO: takePhoto() photoCodec ("hevc" | "jpeg" | "raw")
// TODO: takePhoto() qualityPrioritization
// TODO: takePhoto() enableAutoRedEyeReduction
// TODO: takePhoto() enableAutoStabilization
// TODO: takePhoto() enableAutoDistortionCorrection
// TODO: takePhoto() return with jsi::Value Image reference for faster capture

@Suppress("KotlinJniMissingFunction") // I use fbjni, Android Studio is not smart enough to realize that.
@SuppressLint("ClickableViewAccessibility", "ViewConstructor")
class CameraView(context: Context, private val frameProcessorThread: ExecutorService) :
  ConstraintLayout(context), LifecycleOwner {
  companion object {
    const val TAG = "CameraView"
    const val TAG_PERF = "CameraView.performance"

    private val propsThatRequireSessionReconfiguration = arrayListOf(
      "cameraId",
      "format",
      "fps",
      "hdr",
      "lowLightBoost",
      "photo",
      "video",
      "enableFrameProcessor"
    )
    private val arrayListOfZoom = arrayListOf("zoom")

    @JvmStatic
    var lensFacing = CameraSelector.LENS_FACING_FRONT

//    @JvmStatic
//    var faceDetected: Bitmap? = null
//
//    @JvmStatic
//    var faceDetectedEmbedding: FloatArray? = null
  }

  // react properties
  // props that require reconfiguring
  var cameraId: String? =
    null // this is actually not a react prop directly, but the result of setting device={}
  var enableDepthData = false
  var enableHighQualityPhotos: Boolean? = null
  var enablePortraitEffectsMatteDelivery = false

  // use-cases
  var photo: Boolean? = null
  var video: Boolean? = null
  var audio: Boolean? = null
  var enableFrameProcessor = false

  // props that require format reconfiguring
  var format: ReadableMap? = null
  var fps: Int? = null
  var hdr: Boolean? = null // nullable bool
  var colorSpace: String? = null
  var lowLightBoost: Boolean? = null // nullable bool

  // other props
  var isActive = false
  var torch = "off"
  var zoom: Float = 1f // in "factor"
  var orientation: String? = null
  var enableZoomGesture = false
    set(value) {
      field = value
      setOnTouchListener(if (value) touchEventListener else null)
    }
  var frameProcessorFps = 1.0
    set(value) {
      field = value
      actualFrameProcessorFps = if (value == -1.0) 30.0 else value
      lastFrameProcessorPerformanceEvaluation = System.currentTimeMillis()
      frameProcessorPerformanceDataCollector.clear()
    }

  // private properties
  private var isMounted = false
  private val reactContext: ReactContext
    get() = context as ReactContext

  @Suppress("JoinDeclarationAndAssignment")
  internal val previewView: PreviewView
  private val cameraExecutor = Executors.newSingleThreadExecutor()
  internal val takePhotoExecutor = Executors.newSingleThreadExecutor()
  internal val recordVideoExecutor = Executors.newSingleThreadExecutor()
  internal var coroutineScope = CoroutineScope(Dispatchers.Main)

  internal var camera: Camera? = null
  internal var imageCapture: ImageCapture? = null
  internal var videoCapture: VideoCapture<Recorder>? = null
  private var imageAnalysis: ImageAnalysis? = null
  private var preview: Preview? = null

  internal var activeVideoRecording: Recording? = null

  private var lastFrameProcessorCall = System.currentTimeMillis()

  private var extensionsManager: ExtensionsManager? = null

  private val scaleGestureListener: ScaleGestureDetector.SimpleOnScaleGestureListener
  private val scaleGestureDetector: ScaleGestureDetector
  private val touchEventListener: OnTouchListener

  private val lifecycleRegistry: LifecycleRegistry
  private var hostLifecycleState: Lifecycle.State

  private val inputRotation: Int
    get() {
      return context.displayRotation
    }
  private val outputRotation: Int
    get() {
      if (orientation != null) {
        // user is overriding output orientation
        return when (orientation!!) {
          "portrait" -> Surface.ROTATION_0
          "landscapeRight" -> Surface.ROTATION_90
          "portraitUpsideDown" -> Surface.ROTATION_180
          "landscapeLeft" -> Surface.ROTATION_270
          else -> throw InvalidTypeScriptUnionError("orientation", orientation!!)
        }
      } else {
        // use same as input rotation
        return inputRotation
      }
    }

  private var minZoom: Float = 1f
  private var maxZoom: Float = 1f

  private var actualFrameProcessorFps = 30.0
  private val frameProcessorPerformanceDataCollector = FrameProcessorPerformanceDataCollector()
  private var lastSuggestedFrameProcessorFps = 0.0
  private var lastFrameProcessorPerformanceEvaluation = System.currentTimeMillis()
  private val isReadyForNewEvaluation: Boolean
    get() {
      val lastPerformanceEvaluationElapsedTime =
        System.currentTimeMillis() - lastFrameProcessorPerformanceEvaluation
      return lastPerformanceEvaluationElapsedTime > 1000
    }

  private var needUpdateGraphicOverlayImageSourceInfo = false

  //public static int lensFacing = CameraSelector.LENS_FACING_FRONT
  private var graphicOverlay: GraphicOverlay? = null
  private var imageProcessor: FaceDetectorProcessor? = null
  var compatList = CompatibilityList()

  private val faceRecognitionModel = "ms1m_mobilenetv2_16.tflite"
  private val faceAntiSpoofModel = "AntiSpoofing_bin_1.5_128.onnx"
  var options = Interpreter.Options()

  @DoNotStrip
  private var mHybridData: HybridData? = null

  @Suppress("LiftReturnOrAssignment", "RedundantIf")
  internal val fallbackToSnapshot: Boolean
    @SuppressLint("UnsafeOptInUsageError")
    get() {
      if (video != true && !enableFrameProcessor) {
        // Both use-cases are disabled, so `photo` is the only use-case anyways. Don't need to fallback here.
        return false
      }
      cameraId?.let { cameraId ->
        val cameraManger = reactContext.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        cameraManger?.let {
          val characteristics = cameraManger.getCameraCharacteristics(cameraId)
          val hardwareLevel =
            characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
          if (hardwareLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            // Camera only supports a single use-case at a time
            return true
          } else {
            if (video == true && enableFrameProcessor) {
              // Camera supports max. 2 use-cases, but both are occupied by `frameProcessor` and `video`
              return true
            } else {
              // Camera supports max. 2 use-cases and only one is occupied (either `frameProcessor` or `video`), so we can add `photo`
              return false
            }
          }
        }
      }
      return false
    }

  init {
    if (FrameProcessorRuntimeManager.enableFrameProcessors) {
      mHybridData = initHybrid()
    }
    keepScreenOn = true
    previewView = PreviewView(context)
    previewView.id = generateViewId()
    previewView.layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
    previewView.installHierarchyFitter() // If this is not called correctly, view finder will be black/blank
    addView(previewView)

    graphicOverlay = GraphicOverlay(context, null)
    graphicOverlay?.id = generateViewId()
    graphicOverlay?.layoutParams = LayoutParams(0, 0)
    addView(graphicOverlay)

    val constraintSet = ConstraintSet()
    constraintSet.clone(this@CameraView)
    graphicOverlay?.id?.let {
      constraintSet.connect(
        it,
        ConstraintSet.LEFT,
        previewView.id,
        ConstraintSet.LEFT,
        0
      )
    }
    graphicOverlay?.id?.let {
      constraintSet.connect(
        it,
        ConstraintSet.TOP,
        previewView.id,
        ConstraintSet.TOP,
        0
      )
    }
    graphicOverlay?.id?.let {
      constraintSet.connect(
        it,
        ConstraintSet.BOTTOM,
        previewView.id,
        ConstraintSet.BOTTOM,
        0
      )
    }
    graphicOverlay?.id?.let {
      constraintSet.connect(
        it,
        ConstraintSet.RIGHT,
        previewView.id,
        ConstraintSet.RIGHT,
        0
      )
    }

    var addFaceButton = Button(context)
    var lp = LayoutParams(300, 150);
    addFaceButton.layoutParams = lp;
    addFaceButton.text = "Add Face";
    addFaceButton.id = generateViewId()
    addFaceButton.setOnClickListener {
      val dialog = Dialog(context)
      dialog.requestWindowFeature(1)
      dialog.setContentView(R.layout.add_face_layout)
      dialog.window!!.setBackgroundDrawable(ColorDrawable(0))
      dialog.window!!.setLayout(-1, -2)
      dialog.setCanceledOnTouchOutside(false)
      dialog.setCancelable(false)
      //val faceEmbedding: FloatArray? = faceDetectedEmbedding
      val mName = dialog.findViewById<EditText>(R.id.nameInput)
      val add_face_preview = dialog.findViewById<ImageView>(R.id.faceAddImageView)
      add_face_preview.setImageBitmap(imageProcessor?.faceDetected)

      (dialog.findViewById<View>(R.id.confirmButton) as Button).setOnClickListener {
        FaceDetectorProcessor.persons.add(
          Person(
            mName.text.toString(),
            imageProcessor?.faceDetectedEmbedding!!
          )
        )
        dialog.dismiss()
      }
      (dialog.findViewById<View>(R.id.cancelButton) as Button).setOnClickListener { dialog.dismiss() }

      dialog.show()
    }

    addView(addFaceButton)

//    addFaceButton.id.let {
//      constraintSet.connect(
//        it,
//        ConstraintSet.END,
//        previewView.id,
//        ConstraintSet.END,
//        50
//      )
//    }
//
//    addFaceButton.id.let {
//      constraintSet.connect(
//        it,
//        ConstraintSet.TOP,
//        previewView.id,
//        ConstraintSet.TOP,
//      )
//    }

//    addFaceButton.id.let {
//      constraintSet.connect(
//        it,
//        ConstraintSet.BOTTOM,
//        previewView.id,
//        ConstraintSet.BOTTOM,
//      )
//    }
//
//    addFaceButton.id.let {
//      constraintSet.connect(
//        it,
//        ConstraintSet.START,
//        previewView.id,
//        ConstraintSet.START,
//      )
//    }

//    addFaceButton.id.let {
//      constraintSet.setHorizontalBias(it, 1f)
//    }
//    addFaceButton.id.let {
//      constraintSet.setVerticalBias(it, 0.5f)
//    }

    constraintSet.applyTo(this@CameraView)

    scaleGestureListener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
      override fun onScale(detector: ScaleGestureDetector): Boolean {
        zoom = max(min((zoom * detector.scaleFactor), maxZoom), minZoom)
        update(arrayListOfZoom)
        return true
      }
    }
    scaleGestureDetector = ScaleGestureDetector(context, scaleGestureListener)
    touchEventListener =
      OnTouchListener { _, event -> return@OnTouchListener scaleGestureDetector.onTouchEvent(event) }

    hostLifecycleState = Lifecycle.State.INITIALIZED
    lifecycleRegistry = LifecycleRegistry(this)
    reactContext.addLifecycleEventListener(object : LifecycleEventListener {
      override fun onHostResume() {
        hostLifecycleState = Lifecycle.State.RESUMED
        updateLifecycleState()
        // workaround for https://issuetracker.google.com/issues/147354615, preview must be bound on resume
        update(propsThatRequireSessionReconfiguration)
      }

      override fun onHostPause() {
        hostLifecycleState = Lifecycle.State.CREATED
        updateLifecycleState()
      }

      override fun onHostDestroy() {
        hostLifecycleState = Lifecycle.State.DESTROYED
        updateLifecycleState()
        cameraExecutor.shutdown()
        takePhotoExecutor.shutdown()
        recordVideoExecutor.shutdown()
        reactContext.removeLifecycleEventListener(this)
      }
    })
  }

  override fun onConfigurationChanged(newConfig: Configuration?) {
    super.onConfigurationChanged(newConfig)
    updateOrientation()
  }

  @SuppressLint("RestrictedApi")
  private fun updateOrientation() {
    preview?.targetRotation = inputRotation
    imageCapture?.targetRotation = outputRotation
    videoCapture?.targetRotation = outputRotation
    imageAnalysis?.targetRotation = outputRotation
  }

  private external fun initHybrid(): HybridData
  private external fun frameProcessorCallback(frame: ImageProxy)

  override fun getLifecycle(): Lifecycle {
    return lifecycleRegistry
  }

  /**
   * Updates the custom Lifecycle to match the host activity's lifecycle, and if it's active we narrow it down to the [isActive] and [isAttachedToWindow] fields.
   */
  private fun updateLifecycleState() {
    val lifecycleBefore = lifecycleRegistry.currentState
    if (hostLifecycleState == Lifecycle.State.RESUMED) {
      // Host Lifecycle (Activity) is currently active (RESUMED), so we narrow it down to the view's lifecycle
      if (isActive && isAttachedToWindow) {
        lifecycleRegistry.currentState = Lifecycle.State.RESUMED
      } else {
        lifecycleRegistry.currentState = Lifecycle.State.CREATED
      }
    } else {
      // Host Lifecycle (Activity) is currently inactive (STARTED or DESTROYED), so that overrules our view's lifecycle
      lifecycleRegistry.currentState = hostLifecycleState
    }
    Log.d(
      TAG,
      "Lifecycle went from ${lifecycleBefore.name} -> ${lifecycleRegistry.currentState.name} (isActive: $isActive | isAttachedToWindow: $isAttachedToWindow)"
    )
  }

  override fun onAttachedToWindow() {
    super.onAttachedToWindow()
    updateLifecycleState()
    if (!isMounted) {
      isMounted = true
      invokeOnViewReady()
    }
  }

  override fun onDetachedFromWindow() {
    super.onDetachedFromWindow()
    updateLifecycleState()
  }

  /**
   * Invalidate all React Props and reconfigure the device
   */
  fun update(changedProps: ArrayList<String>) = previewView.post {
    // TODO: Does this introduce too much overhead?
    //  I need to .post on the previewView because it might've not been initialized yet
    //  I need to use CoroutineScope.launch because of the suspend fun [configureSession]
    coroutineScope.launch {
      try {
        val shouldReconfigureSession =
          changedProps.containsAny(propsThatRequireSessionReconfiguration)
        val shouldReconfigureZoom = shouldReconfigureSession || changedProps.contains("zoom")
        val shouldReconfigureTorch = shouldReconfigureSession || changedProps.contains("torch")
        val shouldUpdateOrientation =
          shouldReconfigureSession || changedProps.contains("orientation")

        if (changedProps.contains("isActive")) {
          updateLifecycleState()
        }
        if (shouldReconfigureSession) {
          configureSession()
        }
        if (shouldReconfigureZoom) {
          val zoomClamped = max(min(zoom, maxZoom), minZoom)
          camera!!.cameraControl.setZoomRatio(zoomClamped)
        }
        if (shouldReconfigureTorch) {
          camera!!.cameraControl.enableTorch(torch == "on")
        }
        if (shouldUpdateOrientation) {
          updateOrientation()
        }
      } catch (e: Throwable) {
        Log.e(TAG, "update() threw: ${e.message}")
        invokeOnError(e)
      }
    }
  }

  /**
   * Configures the camera capture session. This should only be called when the camera device changes.
   */
  @SuppressLint("RestrictedApi", "UnsafeOptInUsageError")
  private suspend fun configureSession() {
    try {
      val startTime = System.currentTimeMillis()
      Log.i(TAG, "Configuring session...")
      if (ContextCompat.checkSelfPermission(
          context,
          Manifest.permission.CAMERA
        ) != PackageManager.PERMISSION_GRANTED
      ) {
        throw CameraPermissionError()
      }
      if (cameraId == null) {
        throw NoCameraDeviceError()
      }
      if (format != null)
        Log.i(TAG, "Configuring session with Camera ID $cameraId and custom format...")
      else
        Log.i(TAG, "Configuring session with Camera ID $cameraId and default format options...")

      //Load Face Detector Processsor
      try {
        imageProcessor = FaceDetectorProcessor(context)
      } catch (e: Exception) {
        Log.e(TAG, "Can not create face detect processor")
      }
      //Load Face Recognition Model
      try {
        if (compatList.isDelegateSupportedOnThisDevice) {
          val delegateOptions: GpuDelegateFactory.Options = compatList.bestOptionsForThisDevice
          val gpuDelegate = GpuDelegate(delegateOptions)
          options.addDelegate(gpuDelegate)
          Log.d(TAG, "Running on device's GPU")
        } else {
          options.numThreads = 4
          Log.d(TAG, "Running on device's CPU")
        }
        val inputStream: InputStream = context.assets.open(faceRecognitionModel)
        val model = ByteArray(withContext(Dispatchers.IO) {
          inputStream.available()
        })
        withContext(Dispatchers.IO) {
          inputStream.read(model)
        }
        val buffer = ByteBuffer.allocateDirect(model.size)
          .order(ByteOrder.nativeOrder())
        buffer.put(model)
        FaceDetectorProcessor.tflite = Interpreter(buffer, options)
        Log.d(TAG, "Face Recognition Interpreter Loaded")
      } catch (e: IOException) {
        // File not found?
        Log.e(TAG, "Face Recognition not Loaded $e")
      }

      try {
        FaceDetectorProcessor.env = OrtEnvironment.getEnvironment()
        val opts = SessionOptions()
        opts.setOptimizationLevel(OptLevel.BASIC_OPT)
        val inputStream: InputStream = context.assets.open(faceAntiSpoofModel)
        val model = ByteArray(inputStream.available())
        inputStream.read(model)
        FaceDetectorProcessor.session = FaceDetectorProcessor.env.createSession(model, opts)
        Log.d(TAG, "Face Anti-spoof Session Loaded")
      } catch (e: IOException) {
        // File not found?
        Log.e(TAG, "Face Anti-spoof not Loaded $e")
      } catch (e: OrtException) {
        Log.e(
          TAG,
          "Face Anti-spoof not Loaded $e"
        )
      }
      // Used to bind the lifecycle of cameras to the lifecycle owner
      val cameraProvider = ProcessCameraProvider.getInstance(reactContext).await()

      var cameraSelector = CameraSelector.Builder().byID(cameraId!!).build()

      val tryEnableExtension: (suspend (extension: Int) -> Unit) = lambda@{ extension ->
        if (extensionsManager == null) {
          Log.i(TAG, "Initializing ExtensionsManager...")
          extensionsManager = ExtensionsManager.getInstanceAsync(context, cameraProvider).await()
        }
        if (extensionsManager!!.isExtensionAvailable(cameraSelector, extension)) {
          Log.i(TAG, "Enabling extension $extension...")
          cameraSelector =
            extensionsManager!!.getExtensionEnabledCameraSelector(cameraSelector, extension)
        } else {
          Log.e(TAG, "Extension $extension is not available for the given Camera!")
          throw when (extension) {
            ExtensionMode.HDR -> HdrNotContainedInFormatError()
            ExtensionMode.NIGHT -> LowLightBoostNotContainedInFormatError()
            else -> Error("Invalid extension supplied! Extension $extension is not available.")
          }
        }
      }

      val previewBuilder = Preview.Builder()
        .setTargetRotation(inputRotation)

      val imageCaptureBuilder = ImageCapture.Builder()
        .setTargetName("ImageCapture")
        .setTargetRotation(outputRotation)
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)

      val videoRecorderBuilder = Recorder.Builder()
        .setExecutor(cameraExecutor)

      val imageAnalysisBuilder = ImageAnalysis.Builder()
        .setTargetName("ImageAnalysis")
        .setTargetRotation(outputRotation)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setBackgroundExecutor(frameProcessorThread)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)

      if (format == null) {
        // let CameraX automatically find best resolution for the target aspect ratio
        Log.i(
          TAG,
          "No custom format has been set, CameraX will automatically determine best configuration..."
        )
        val aspectRatio = aspectRatio(
          previewView.height,
          previewView.width
        ) // flipped because it's in sensor orientation.
        previewBuilder.setTargetAspectRatio(aspectRatio)
        imageCaptureBuilder.setTargetAspectRatio(aspectRatio)
        // TODO: Aspect Ratio for Video Recorder?
        imageAnalysisBuilder.setTargetAspectRatio(aspectRatio)
      } else {
        // User has selected a custom format={}. Use that
        val format = DeviceFormat(format!!)
        Log.i(
          TAG,
          "Using custom format - photo: ${format.photoSize}, video: ${format.videoSize} @ $fps FPS"
        )
        if (video == true) {
          previewBuilder.setTargetResolution(format.videoSize)
        } else {
          previewBuilder.setTargetResolution(format.photoSize)
        }
        imageCaptureBuilder.setTargetResolution(format.photoSize)
        imageAnalysisBuilder.setTargetResolution(format.photoSize)

        // TODO: Ability to select resolution exactly depending on format? Just like on iOS...
        when (min(format.videoSize.height, format.videoSize.width)) {
          in 0..480 -> videoRecorderBuilder.setQualitySelector(QualitySelector.from(Quality.SD))
          in 480..720 -> videoRecorderBuilder.setQualitySelector(
            QualitySelector.from(
              Quality.HD,
              FallbackStrategy.lowerQualityThan(Quality.HD)
            )
          )
          in 720..1080 -> videoRecorderBuilder.setQualitySelector(
            QualitySelector.from(
              Quality.FHD,
              FallbackStrategy.lowerQualityThan(Quality.FHD)
            )
          )
          in 1080..2160 -> videoRecorderBuilder.setQualitySelector(
            QualitySelector.from(
              Quality.UHD,
              FallbackStrategy.lowerQualityThan(Quality.UHD)
            )
          )
          in 2160..4320 -> videoRecorderBuilder.setQualitySelector(
            QualitySelector.from(
              Quality.HIGHEST,
              FallbackStrategy.lowerQualityThan(Quality.HIGHEST)
            )
          )
        }

        fps?.let { fps ->
          if (format.frameRateRanges.any { it.contains(fps) }) {
            // Camera supports the given FPS (frame rate range)
            val frameDuration = (1.0 / fps.toDouble()).toLong() * 1_000_000_000

            Log.i(
              TAG,
              "Setting AE_TARGET_FPS_RANGE to $fps-$fps, and SENSOR_FRAME_DURATION to $frameDuration"
            )
            Camera2Interop.Extender(previewBuilder)
              .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(fps, fps))
              .setCaptureRequestOption(CaptureRequest.SENSOR_FRAME_DURATION, frameDuration)
            // TODO: Frame Rate/FPS for Video Recorder?
          } else {
            throw FpsNotContainedInFormatError(fps)
          }
        }
        if (hdr == true) {
          tryEnableExtension(ExtensionMode.HDR)
        }
        if (lowLightBoost == true) {
          tryEnableExtension(ExtensionMode.NIGHT)
        }
      }


      // Unbind use cases before rebinding
      videoCapture = null
      imageCapture = null
      imageAnalysis = null
      cameraProvider.unbindAll()

      // Bind use cases to camera
      val useCases = ArrayList<UseCase>()
//      if (video == true) {
//        Log.i(TAG, "Adding VideoCapture use-case...")
//        Toast.makeText(
//          graphicOverlay!!.context,
//          "Adding VideoCapture use-case...",
//          Toast.LENGTH_SHORT
//        ).show()
//        val videoRecorder = videoRecorderBuilder.build()
//        videoCapture = VideoCapture.withOutput(videoRecorder)
//        videoCapture!!.targetRotation = outputRotation
//        useCases.add(videoCapture!!)
//      }
      if (photo == true) {
        if (fallbackToSnapshot) {
          Log.i(
            TAG,
            "Tried to add photo use-case (`photo={true}`) but the Camera device only supports " +
              "a single use-case at a time. Falling back to Snapshot capture."
          )
        } else {
          Log.i(TAG, "Adding ImageCapture use-case...")
          imageCapture = imageCaptureBuilder.build()
          useCases.add(imageCapture!!)
        }
      }
      Log.i(TAG, "Adding ImageAnalysis use-case...")
      needUpdateGraphicOverlayImageSourceInfo = true
      imageAnalysis = imageAnalysisBuilder.build().apply {
        setAnalyzer(cameraExecutor) { imageProxy ->
          val now = System.currentTimeMillis()
          val intervalMs = (1.0 / actualFrameProcessorFps) * 1000.0
          if (now - lastFrameProcessorCall > intervalMs) {
            lastFrameProcessorCall = now

            val perfSample =
              frameProcessorPerformanceDataCollector.beginPerformanceSampleCollection()
            frameProcessorCallback(imageProxy)
            perfSample.endPerformanceSampleCollection()
          }
          //imageProxy.close()

          if (isReadyForNewEvaluation) {
            // last evaluation was more than a second ago, evaluate again
            evaluateNewPerformanceSamples()
          }
          if (needUpdateGraphicOverlayImageSourceInfo) {
            val isImageFlipped =
              lensFacing == CameraSelector.LENS_FACING_FRONT
            val rotationDegrees: Int = imageProxy.imageInfo.rotationDegrees
            if (rotationDegrees == 0 || rotationDegrees == 180) {
              graphicOverlay?.setImageSourceInfo(
                imageProxy.width, imageProxy.height, isImageFlipped
              )
            } else {
              graphicOverlay?.setImageSourceInfo(
                imageProxy.height, imageProxy.width, isImageFlipped
              )
            }
            needUpdateGraphicOverlayImageSourceInfo = false
          }
          try {
            imageProcessor?.processImageProxy(imageProxy, graphicOverlay)
            val rotateDegree: Int = imageProxy.imageInfo.rotationDegrees
            if (FaceDetectorProcessor.faceDetectProcessor) {
              FaceDetectorProcessor.image = rotateBitmap(
                toBitmap(imageProxy.image)!!,
                rotateDegree,
                flipX = false,
                flipY = false
              )
            }
          } catch (e: MlKitException) {
            Log.e(
              TAG,
              "Failed to process image. Error: " + e.localizedMessage
            )
          }
        }
      }
      useCases.add(imageAnalysis!!)

      preview = previewBuilder.build()
      Log.i(TAG, "Attaching ${useCases[0].name} use-cases...")
      camera =
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, *useCases.toTypedArray())
      preview!!.setSurfaceProvider(previewView.surfaceProvider)

      minZoom = camera!!.cameraInfo.zoomState.value?.minZoomRatio ?: 1f
      maxZoom = camera!!.cameraInfo.zoomState.value?.maxZoomRatio ?: 1f

      val duration = System.currentTimeMillis() - startTime
      Log.i(TAG_PERF, "Session configured in $duration ms! Camera: ${camera!!}")
      invokeOnInitialized()
    } catch (exc: Throwable) {
      Log.e(TAG, "Failed to configure session: ${exc.message}")
      throw when (exc) {
        is CameraError -> exc
        is IllegalArgumentException -> {
          if (exc.message?.contains("too many use cases") == true) {
            ParallelVideoProcessingNotSupportedError(exc)
          } else {
            InvalidCameraDeviceError(exc)
          }
        }
        else -> UnknownCameraError(exc)
      }
    }
  }

  private fun evaluateNewPerformanceSamples() {
    lastFrameProcessorPerformanceEvaluation = System.currentTimeMillis()
    val maxFrameProcessorFps = 30 // TODO: Get maxFrameProcessorFps from ImageAnalyser
    val averageFps = 1.0 / frameProcessorPerformanceDataCollector.averageExecutionTimeSeconds
    val suggestedFrameProcessorFps = floor(min(averageFps, maxFrameProcessorFps.toDouble()))

    if (frameProcessorFps == -1.0) {
      // frameProcessorFps="auto"
      actualFrameProcessorFps = suggestedFrameProcessorFps
    } else {
      // frameProcessorFps={someCustomFpsValue}
      if (suggestedFrameProcessorFps != lastSuggestedFrameProcessorFps && suggestedFrameProcessorFps != frameProcessorFps) {
        invokeOnFrameProcessorPerformanceSuggestionAvailable(
          frameProcessorFps,
          suggestedFrameProcessorFps
        )
        lastSuggestedFrameProcessorFps = suggestedFrameProcessorFps
      }
    }
  }

  private fun rotateBitmap(
    bitmap: Bitmap,
    rotationDegrees: Int,
    flipX: Boolean,
    flipY: Boolean
  ): Bitmap? {
    val matrix = Matrix()

    // Rotate the image back to straight.
    matrix.postRotate(rotationDegrees.toFloat())

    // Mirror the image along the X or Y axis.
    matrix.postScale(if (flipX) -1.0f else 1.0f, if (flipY) -1.0f else 1.0f)
    val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

    // Recycle the old bitmap if it has changed.
    if (rotatedBitmap != bitmap) {
      bitmap.recycle()
    }
    return rotatedBitmap
  }

  private fun toBitmap(image: Image?): Bitmap? {
    val nv21: ByteArray? = YUV_420_888toNV21(image!!)
    /*return rgbToBitmap(decodeYUV420SPtoRGB(nv21,image.getWidth(), image.getHeight()), image.getWidth(), image.getHeight());*/
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
  }

  private fun YUV_420_888toNV21(image: Image): ByteArray? {
    val width = image.width
    val height = image.height
    val ySize = width * height
    val uvSize = width * height / 4
    val nv21 = ByteArray(ySize + uvSize * 2)
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer
    var rowStride = image.planes[0].rowStride
    assert(image.planes[0].pixelStride == 1)
    var pos = 0
    if (rowStride == width) {
      yBuffer[nv21, 0, ySize]
      pos += ySize
    } else {
      var yBufferPos = -rowStride.toLong()
      while (pos < ySize) {
        yBufferPos += rowStride.toLong()
        yBuffer.position(yBufferPos.toInt())
        yBuffer[nv21, pos, width]
        pos += width
      }
    }
    rowStride = image.planes[2].rowStride
    val pixelStride = image.planes[2].pixelStride
    assert(rowStride == image.planes[1].rowStride)
    assert(pixelStride == image.planes[1].pixelStride)
    if (pixelStride == 2 && rowStride == width && uBuffer[0] == vBuffer[1]) {
      val savePixel = vBuffer[1]
      try {
        vBuffer.put(1, savePixel.inv())
        if (uBuffer[0] == savePixel.inv()) {
          vBuffer.put(1, savePixel)
          vBuffer.position(0)
          uBuffer.position(0)
          vBuffer[nv21, ySize, 1]
          uBuffer[nv21, ySize + 1, uBuffer.remaining()]
          return nv21
        }
      } catch (ignored: ReadOnlyBufferException) {
      }
      vBuffer.put(1, savePixel)
    }
    for (row in 0 until height / 2) {
      for (col in 0 until width / 2) {
        val vuPos = col * pixelStride + row * rowStride
        nv21[pos++] = vBuffer[vuPos]
        nv21[pos++] = uBuffer[vuPos]
      }
    }
    return nv21
  }
}
