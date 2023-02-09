package com.mrousavy.camera.frameprocessor;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;
import com.mrousavy.camera.CameraView;
import com.mrousavy.camera.helpers.FaceGraphic;
import com.mrousavy.camera.helpers.GraphicOverlay;
import com.mrousavy.camera.models.Person;

import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {

    private static final String TAG = "FaceDetectorProcessor";

    public final FaceDetector detector;
    public static boolean faceDetectProcessor;

    public static Bitmap image;
    public static Interpreter tflite;

    public static OrtEnvironment env;
    public static OrtSession session;

    private ByteBuffer imgData; // for Quantized model
    public String name;
    public static ArrayList<Person> persons = new ArrayList<>();
    private final float spoofThreshold = 0.2f;
    private final float similarityThreshold = 0.8f;
    private final boolean QUANTIZATION = false;
    //Float model
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    private static final int inputSize = 112;
    private static final int antiSpoofInputSize = 128;
    //Output size of model
    private static final int outputSize = 512;

    public float[] faceDetectedEmbedding;

    public Bitmap faceDetected;

    private Context context;

    public FaceDetectorProcessor(Context context) {
        super(context);
        this.context = context;
        faceDetectProcessor = true;
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setMinFaceSize(0.1f).build();
        detector = FaceDetection.getClient(faceDetectorOptions);

        int numBytesPerChannel;
        if (QUANTIZATION) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        imgData = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * numBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
        faceDetectProcessor = false;
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected Face
    onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay) {
        for (Face face : faces) {
            /** Extract Face Bitmap **/
            Rect facePos = face.getBoundingBox();
            float[] mirrorY = {
                    -1, 0, 0,
                    0, 1, 0,
                    0, 0, 1
            };
            try {
                String result;
                Matrix matrix = new Matrix();
                matrix.setValues(mirrorY);

                Bitmap faceBitmap = Bitmap.createBitmap(
                        image,
                        facePos.left, facePos.top,
                        facePos.width(), facePos.height(),
                        matrix,
                        false);

                Bitmap spoofFaceBitmap = Bitmap.createBitmap(
                        image,
                        facePos.left - 80, facePos.top - 100,
                        facePos.width() + 160, facePos.height() + 150,
                        matrix,
                        false);

                Bitmap scaledFaceBitmap = Bitmap.createScaledBitmap(faceBitmap, inputSize, inputSize, true);

                if (CameraView.getLensFacing() == CameraSelector.LENS_FACING_FRONT) {
                    //CameraXLivePreviewActivity.face_preview.setImageBitmap(scaledFaceBitmap);
                    //CameraView.setFaceDetected(scaledFaceBitmap);
                    faceDetected = scaledFaceBitmap;
                } else {
                    //CameraXLivePreviewActivity.face_preview.setImageBitmap(flippedBitmap(scaledFaceBitmap, true, false));
                    //CameraView.setFaceDetected(flippedBitmap(scaledFaceBitmap, true, false));
                    faceDetected = flippedBitmap(scaledFaceBitmap, true, false);
                }

                byte[][] output_int = new byte[1][outputSize];
                float[][] output = new float[1][outputSize];
                if (QUANTIZATION) {
                    tflite.run(preProcess(scaledFaceBitmap), output_int);
                } else {
                    tflite.run(preProcess(scaledFaceBitmap), output);
                }

                float[] norm_out = new float[outputSize];
                if (QUANTIZATION) {
                    double norm = norm(output_int[0]);
                    for (int i = 0; i < output_int[0].length; i++) {
                        int uint8 = output_int[0][i] & 0xFF;
                        norm_out[i] = (float) (uint8 / norm);
                    }
                } else {
                    double norm = norm(output[0]);
                    for (int i = 0; i < output[0].length; i++) {
                        norm_out[i] = (float) (output[0][i] / norm);
                    }
                }

                //CameraView.setFaceDetectedEmbedding(norm_out);
                faceDetectedEmbedding = norm_out;
                //Check Face Spoofing
                if (checkLuminousity(spoofFaceBitmap)) {
                    //if(face.getHeadEulerAngleY() > -3 && face.getHeadEulerAngleY() < 3 && face.getHeadEulerAngleZ() > -3 && face.getHeadEulerAngleZ() < 3){
                    if (checkAntiSpoof(antiSpoofPreProcess(spoofFaceBitmap))) {
                        //Face Recognition
                        result = recognize(norm_out);
                    } else result = "Face spoofing detected";
                    //} else result = "Please look straight at the camera";
                    graphicOverlay.add(new FaceGraphic(graphicOverlay, face, result));
                }
            } catch (IllegalArgumentException | OrtException e) {
                Log.e(TAG, "java.lang.IllegalArgumentException");
                e.printStackTrace();
            }
            //logExtrasForTesting(face);
        }
        return null;
    }

    private boolean checkAntiSpoof(ByteBuffer antiSpoofImgData) throws OrtException {
        FloatBuffer convertedAntiSpoofImgData = ((ByteBuffer) antiSpoofImgData.rewind()).asFloatBuffer();

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        long[] shape = {1, 3, antiSpoofInputSize, antiSpoofInputSize};
        inputMap.put(session.getInputNames().iterator().next(), OnnxTensor.createTensor(env, convertedAntiSpoofImgData, shape));
        OrtSession.Result antiSpoof = session.run(inputMap);

        float[][] tmp = (float[][]) antiSpoof.get(0).getValue();
        //float[] result = {tmp[0][0], tmp[0][1], tmp[0][2]};
        float[] result = {tmp[0][0], tmp[0][1]};
        Log.d(TAG, "Antispoof Result: " + Arrays.toString(result));
        float[] as_result = softMax(result);
        float antiSpoofScore = as_result[0];
        int prediction = argMax(as_result);

        Log.d(TAG, "Antispoof Raw: " + Arrays.deepToString(tmp));
        Log.d(TAG, "Antispoof Softmax: " + Arrays.toString(as_result));
        Log.d(TAG, "Antispoof Argmax: " + prediction);
        Log.d(TAG, "Antispoof Score: " + antiSpoofScore);

        if (antiSpoofScore > spoofThreshold) {
            //if (prediction == 0) {
            Log.d(TAG, "Face is real");
            return true;
        } else {
            Log.d(TAG, "Face is spoof");
            return false;
        }
    }

    public int calculateBrightnessEstimate(Bitmap bitmap, int pixelSpacing) {
        int R = 0;
        int G = 0;
        int B = 0;
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        int n = 0;
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < pixels.length; i += pixelSpacing) {
            int color = pixels[i];
            R += Color.red(color);
            G += Color.green(color);
            B += Color.blue(color);
            n++;
        }
        return (R + B + G) / (n * 3);
    }

    public boolean checkLuminousity(Bitmap bitmap) {
        int luminousity = calculateBrightnessEstimate(bitmap, 1);
        Log.d(TAG, "Luminousity: " + luminousity);
        //if(luminousity >= 140) return false;
        //else return true;
        return true;
    }

    private ByteBuffer antiSpoofPreProcess(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4 * antiSpoofInputSize * antiSpoofInputSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[antiSpoofInputSize * antiSpoofInputSize];

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, antiSpoofInputSize, antiSpoofInputSize, true);
        resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, false);
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < antiSpoofInputSize; ++i) {
            for (int j = 0; j < antiSpoofInputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    private ByteBuffer preProcess(Bitmap faceBitmap) {
        imgData.rewind();
        int[] face_pix = new int[inputSize * inputSize];
        faceBitmap.getPixels(face_pix, 0, inputSize, 0, 0, inputSize, inputSize);
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                int index = y * inputSize + x;
                if (QUANTIZATION) {
                    imgData.put((byte) ((face_pix[index] >> 16) & 0xFF));
                    imgData.put((byte) ((face_pix[index] >> 8) & 0xFF));
                    imgData.put((byte) (face_pix[index] & 0xFF));
                } else {
                    imgData.putFloat((((face_pix[index] >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((face_pix[index] >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((face_pix[index] & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    public static int argMax(float[] inputs) {
        int maxIndex = 0;
        for (int i = 1; i < inputs.length; i++) {
            if (inputs[i] > inputs[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static float[] softMax(float[] inputs) {
        float sum = 0;
        float[] result = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = (float) Math.exp(inputs[i]);
            sum += result[i];
        }
        for (int i = 0; i < inputs.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    private Bitmap flippedBitmap(Bitmap source, boolean xFlip, boolean yFlip) {
        Matrix matrix = new Matrix();
        matrix.postScale(xFlip ? -1 : 1, yFlip ? -1 : 1, source.getWidth() / 2f, source.getHeight() / 2f);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    @SuppressLint("DefaultLocale")
    private String recognize(float[] embedding) {
        if (!persons.isEmpty()) {
            float maxVal = 0;
            for (Person person : persons) {
                Log.d(TAG, "Person name: " + person.getName());
                Log.d(TAG, "Person embed: " + Arrays.toString(person.getEmbedding()));
                if (cosineSimilarity(person.getEmbedding(), embedding) > maxVal) {
                    maxVal = cosineSimilarity(person.getEmbedding(), embedding);
                    name = person.getName();
                }
            }

            if (maxVal > similarityThreshold) {
                //Ca.recognize_name.setText(name);

                return "Name: " + name + "\nConfidence: " + String.format("%.2f", maxVal * 100) + "%";
            } else {
                return "Stranger";
            }
        } else {
            return "Stranger";
        }
    }

    private static float cosineSimilarity(float[] vectorA, float[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    private static void logExtrasForTesting(Face face) {
        if (face != null) {
            // All landmarks
            int[] landMarkTypes =
                    new int[]{
                            FaceLandmark.MOUTH_BOTTOM,
                            FaceLandmark.MOUTH_RIGHT,
                            FaceLandmark.MOUTH_LEFT,
                            FaceLandmark.RIGHT_EYE,
                            FaceLandmark.LEFT_EYE,
                            FaceLandmark.RIGHT_EAR,
                            FaceLandmark.LEFT_EAR,
                            FaceLandmark.RIGHT_CHEEK,
                            FaceLandmark.LEFT_CHEEK,
                            FaceLandmark.NOSE_BASE
                    };
            String[] landMarkTypesStrings =
                    new String[]{
                            "MOUTH_BOTTOM",
                            "MOUTH_RIGHT",
                            "MOUTH_LEFT",
                            "RIGHT_EYE",
                            "LEFT_EYE",
                            "RIGHT_EAR",
                            "LEFT_EAR",
                            "RIGHT_CHEEK",
                            "LEFT_CHEEK",
                            "NOSE_BASE"
                    };
            for (int i = 0; i < landMarkTypes.length; i++) {
                FaceLandmark landmark = face.getLandmark(landMarkTypes[i]);
                if (landmark == null) {
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "No landmark of type: " + landMarkTypesStrings[i] + " has been detected");
                } else {
                    PointF landmarkPosition = landmark.getPosition();
                    String landmarkPositionStr =
                            String.format(Locale.US, "x: %f , y: %f", landmarkPosition.x, landmarkPosition.y);
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "Position for face landmark: "
                                    + landMarkTypesStrings[i]
                                    + " is :"
                                    + landmarkPositionStr);
                }
            }
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face left eye open probability: " + face.getLeftEyeOpenProbability());
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face right eye open probability: " + face.getRightEyeOpenProbability());
            Log.v(MANUAL_TESTING_LOG, "face smiling probability: " + face.getSmilingProbability());
            Log.v(MANUAL_TESTING_LOG, "face tracking id: " + face.getTrackingId());
        }
    }

    public static double norm(byte[] data) {
        return (Math.sqrt(sumSquares(data)));
    }

    public static double norm(float[] data) {
        return (Math.sqrt(sumSquares(data)));
    }

    public static int sumSquares(byte[] data) {
        int ans = 0;
        for (int k = 0; k < data.length; k++) {
            int uint8 = data[k] & 0xFF;
            ans += uint8 * uint8;
        }
        return (ans);
    }

    public static float sumSquares(float[] data) {
        float ans = 0.0f;
        for (int k = 0; k < data.length; k++) {
            ans += data[k] * data[k];
        }
        return (ans);
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }
}
