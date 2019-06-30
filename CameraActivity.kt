/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification

import android.Manifest
import android.app.Fragment
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Camera
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.Image
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.media.MediaPlayer
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Trace
import androidx.annotation.UiThread
import com.google.android.material.bottomsheet.BottomSheetBehavior
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.ViewTreeObserver
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.MediaController
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import java.nio.ByteBuffer
import org.tensorflow.lite.examples.classification.env.ImageUtils
import org.tensorflow.lite.examples.classification.env.Logger
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model
import org.tensorflow.lite.examples.classification.tflite.Classifier.Recognition

abstract class CameraActivity : AppCompatActivity(), OnImageAvailableListener, Camera.PreviewCallback, View.OnClickListener, AdapterView.OnItemSelectedListener {
    protected var previewWidth = 0
    protected var previewHeight = 0
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var useCamera2API: Boolean = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    protected var luminanceStride: Int = 0
        private set
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private var bottomSheetLayout: LinearLayout? = null
    private var gestureLayout: LinearLayout? = null
    private var sheetBehavior: BottomSheetBehavior? = null
    protected var recognitionTextView: TextView
    protected var recognition1TextView: TextView
    protected var recognition2TextView: TextView
    protected var recognitionValueTextView: TextView
    protected var recognition1ValueTextView: TextView
    protected var recognition2ValueTextView: TextView
    protected var FinalView: TextView
    protected var frameValueTextView: TextView
    protected var cropValueTextView: TextView
    protected var cameraResolutionTextView: TextView
    protected var rotationTextView: TextView
    protected var inferenceTimeTextView: TextView
    protected var bottomSheetArrowImageView: ImageView
    private var plusImageView: ImageView? = null
    private var minusImageView: ImageView? = null
    private var modelSpinner: Spinner? = null
    private var deviceSpinner: Spinner? = null
    private var threadsTextView: TextView? = null

    private var model = Model.QUANTIZED
    private var device = Device.CPU
    private var numThreads = -1

    protected val luminance: ByteArray
        get() = yuvBytes[0]

    protected val screenOrientation: Int
        get() {
            when (getWindowManager().getDefaultDisplay().getRotation()) {
                Surface.ROTATION_270 -> return 270
                Surface.ROTATION_180 -> return 180
                Surface.ROTATION_90 -> return 90
                else -> return 0
            }
        }

    protected abstract val layoutId: Int

    protected abstract val desiredPreviewFrameSize: Size


    @Override
    protected fun onCreate(savedInstanceState: Bundle) {
        LOGGER.d("onCreate $this")
        super.onCreate(null)
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_camera)
        val toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        getSupportActionBar().setDisplayShowTitleEnabled(false)

        if (hasPermission()) {
            setFragment()
        } else {
            requestPermission()
        }

        threadsTextView = findViewById(R.id.threads)
        plusImageView = findViewById(R.id.plus)
        minusImageView = findViewById(R.id.minus)
        modelSpinner = findViewById(R.id.model_spinner)
        deviceSpinner = findViewById(R.id.device_spinner)
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout)
        gestureLayout = findViewById(R.id.gesture_layout)
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout)
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow)

        val vto = gestureLayout!!.getViewTreeObserver()
        vto.addOnGlobalLayoutListener(
                object : ViewTreeObserver.OnGlobalLayoutListener() {
                    @Override
                    fun onGlobalLayout() {
                        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                            gestureLayout!!.getViewTreeObserver().removeGlobalOnLayoutListener(this)
                        } else {
                            gestureLayout!!.getViewTreeObserver().removeOnGlobalLayoutListener(this)
                        }
                        //                int width = bottomSheetLayout.getMeasuredWidth();
                        //int height = gestureLayout.getMeasuredHeight();

                        sheetBehavior!!.setPeekHeight(0)
                    }
                })
        sheetBehavior!!.setHideable(false)

        sheetBehavior!!.setBottomSheetCallback(
                object : BottomSheetBehavior.BottomSheetCallback() {
                    @Override
                    fun onStateChanged(@NonNull bottomSheet: View, newState: Int) {
                        when (newState) {
                            BottomSheetBehavior.STATE_HIDDEN -> {
                            }
                            BottomSheetBehavior.STATE_EXPANDED -> {
                                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down)
                            }
                            BottomSheetBehavior.STATE_COLLAPSED -> {
                                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up)
                            }
                            BottomSheetBehavior.STATE_DRAGGING -> {
                            }
                            BottomSheetBehavior.STATE_SETTLING -> bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up)
                        }
                    }

                    @Override
                    fun onSlide(@NonNull bottomSheet: View, slideOffset: Float) {
                    }
                })

        recognitionTextView = findViewById(R.id.detected_item)
        FinalView = findViewById(R.id.Final)
        recognitionValueTextView = findViewById(R.id.detected_item_value)
        recognition1TextView = findViewById(R.id.detected_item1)
        recognition1ValueTextView = findViewById(R.id.detected_item1_value)
        recognition2TextView = findViewById(R.id.detected_item2)
        recognition2ValueTextView = findViewById(R.id.detected_item2_value)

        frameValueTextView = findViewById(R.id.frame_info)
        cropValueTextView = findViewById(R.id.crop_info)
        cameraResolutionTextView = findViewById(R.id.view_info)
        rotationTextView = findViewById(R.id.rotation_info)
        inferenceTimeTextView = findViewById(R.id.inference_info)

        modelSpinner!!.setOnItemSelectedListener(this)
        deviceSpinner!!.setOnItemSelectedListener(this)

        plusImageView!!.setOnClickListener(this)
        minusImageView!!.setOnClickListener(this)

        model = Model.valueOf(modelSpinner!!.getSelectedItem().toString().toUpperCase())
        device = Device.valueOf(deviceSpinner!!.getSelectedItem().toString())
        numThreads = Integer.parseInt(threadsTextView!!.getText().toString().trim())
    }

    protected fun getRgbBytes(): IntArray? {
        imageConverter!!.run()
        return rgbBytes
    }

    /** Callback for android.hardware.Camera API  */
    @Override
    fun onPreviewFrame(bytes: ByteArray, camera: Camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!")
            return
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                val previewSize = camera.getParameters().getPreviewSize()
                previewHeight = previewSize.height
                previewWidth = previewSize.width
                rgbBytes = IntArray(previewWidth * previewHeight)
                onPreviewSizeChosen(Size(previewSize.width, previewSize.height), 90)
            }
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            return
        }

        isProcessingFrame = true
        yuvBytes[0] = bytes
        luminanceStride = previewWidth

        imageConverter = object : Runnable() {
            @Override
            fun run() {
                ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes)
            }
        }

        postInferenceCallback = object : Runnable() {
            @Override
            fun run() {
                camera.addCallbackBuffer(bytes)
                isProcessingFrame = false
            }
        }
        processImage()
    }

    /** Callback for Camera2 API  */
    @Override
    fun onImageAvailable(reader: ImageReader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return

            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            Trace.beginSection("imageAvailable")
            val planes = image.getPlanes()
            fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].getRowStride()
            val uvRowStride = planes[1].getRowStride()
            val uvPixelStride = planes[1].getPixelStride()

            imageConverter = object : Runnable() {
                @Override
                fun run() {
                    ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0],
                            yuvBytes[1],
                            yuvBytes[2],
                            previewWidth,
                            previewHeight,
                            luminanceStride,
                            uvRowStride,
                            uvPixelStride,
                            rgbBytes)
                }
            }

            postInferenceCallback = object : Runnable() {
                @Override
                fun run() {
                    image.close()
                    isProcessingFrame = false
                }
            }

            processImage()
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            Trace.endSection()
            return
        }

        Trace.endSection()
    }

    @Override
    @Synchronized
    fun onStart() {
        LOGGER.d("onStart $this")
        super.onStart()
    }

    @Override
    @Synchronized
    fun onResume() {
        LOGGER.d("onResume $this")
        super.onResume()

        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.getLooper())
    }

    @Override
    @Synchronized
    fun onPause() {
        LOGGER.d("onPause $this")

        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }

        super.onPause()
    }

    @Override
    @Synchronized
    fun onStop() {
        LOGGER.d("onStop $this")
        super.onStop()
    }

    @Override
    @Synchronized
    fun onDestroy() {
        LOGGER.d("onDestroy $this")
        super.onDestroy()
    }

    @Synchronized
    protected fun runInBackground(r: Runnable) {
        if (handler != null) {
            handler!!.post(r)
        }
    }

    @Override
    fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.size > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                setFragment()
            } else {
                requestPermission()
            }
        }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            checkSelfPermission(PERMISSION_CAMERA) === PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        this@CameraActivity,
                        "Camera permission is required for this app",
                        Toast.LENGTH_LONG)
                        .show()
            }
            requestPermissions(arrayOf<String>(PERMISSION_CAMERA), PERMISSIONS_REQUEST)
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
            characteristics: CameraCharacteristics, requiredLevel: Int): Boolean {
        val deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
    }

    private fun chooseCamera(): String? {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.getCameraIdList()) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing === CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }

                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue

// Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = facing === CameraCharacteristics.LENS_FACING_EXTERNAL || isHardwareLevelSupported(
                        characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL)
                LOGGER.i("Camera API lv2?: %s", useCamera2API)
                return cameraId
            }
        } catch (e: CameraAccessException) {
            LOGGER.e(e, "Not allowed to access camera")
        }

        return null
    }

    protected fun setFragment() {
        val cameraId = chooseCamera()

        val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment = CameraConnectionFragment.newInstance(
                    object : CameraConnectionFragment.ConnectionCallback() {
                        @Override
                        fun onPreviewSizeChosen(size: Size, rotation: Int) {
                            previewHeight = size.getHeight()
                            previewWidth = size.getWidth()
                            this@CameraActivity.onPreviewSizeChosen(size, rotation)
                        }
                    },
                    this,
                    layoutId,
                    desiredPreviewFrameSize)

            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment = LegacyCameraConnectionFragment(this, layoutId, desiredPreviewFrameSize)
        }

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit()
    }

    protected fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].getBuffer()
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity())
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer.get(yuvBytes[i])
        }
    }

    protected fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }

    @UiThread
    protected fun showResultsInBottomSheet(results: List<Recognition>?) {


        if (results != null && results.size() >= 3) {
            val recognition = results[0]
            if (recognition != null) {
                if (recognition!!.getTitle() != null) {

                    // the final result will be displayed from here.
                    val s1 = recognition!!.getTitle()
                    if (s1.equals("100_new_back") || s1.equals("100_new_front") || s1.equals("100_old_back") || s1.equals("100_old_back")) {
                        FinalView.setText("₹100")
                        val mp100 = MediaPlayer.create(this, R.raw.a100)
                        val tv = findViewById(R.id.Final)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp100.start()
                            }
                        })


                    } else if (s1.equals("10_new_back") || s1.equals("10_new_front") || s1.equals("10_old_back") || s1.equals("10_old_back")) {
                        FinalView.setText("₹10")
                        val mp10 = MediaPlayer.create(this, R.raw.a10)
                        val tv = findViewById(R.id.Final)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp10.start()
                            }
                        })

                    } else if (s1.equals("2000_back") || s1.equals("2000_front")) {
                        FinalView.setText("₹2000")
                        val tv = findViewById(R.id.Final)
                        val mp2000 = MediaPlayer.create(this, R.raw.a2000)

                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp2000.start()
                            }
                        })

                    } else if (s1.equals("200_back") || s1.equals("200_front")) {
                        FinalView.setText("₹200")
                        val tv = findViewById(R.id.Final)
                        val mp200 = MediaPlayer.create(this, R.raw.a200)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp200.start()
                            }
                        })

                    } else if (s1.equals("20_old_back") || s1.equals("20_old_front")) {
                        FinalView.setText("₹20")
                        val tv = findViewById(R.id.Final)
                        val mp20 = MediaPlayer.create(this, R.raw.a20)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp20.start()
                            }
                        })

                    } else if (s1.equals("500_new_back") || s1.equals("500_new_front")) {
                        FinalView.setText("₹500")
                        ///# RIP 500 old :-P
                        val tv = findViewById(R.id.Final)
                        val mp500 = MediaPlayer.create(this, R.raw.a500)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp500.start()
                            }
                        })
                    } else if (s1.equals("50_new_back") || s1.equals("50_new_front") || s1.equals("50_old_back") || s1.equals("50_old_back")) {
                        FinalView.setText("₹50")
                        val tv = findViewById(R.id.Final)
                        val mp50 = MediaPlayer.create(this, R.raw.a50)
                        tv.setOnClickListener(object : View.OnClickListener() {
                            @Override
                            fun onClick(v: View) {
                                mp50.start()
                            }
                        })

                    }
                    // FinalView.setText(recognition.getTitle()) yaha se change hoga;

                    //recognitionTextView.setText(recognition.getTitle());}

                    // if (recognition.getConfidence() != null)
                    // recognitionValueTextView.setText(
                    //String.format("%.2f", (100 * recognition.getConfidence())) + "%");
                }

                //ADVANCED
                // Recognition recognition1 = results.get(1);
                // if (recognition1 != null) {
                //   if (recognition1.getTitle() != null) recognition1TextView.setText(recognition1.getTitle());
                // if (recognition1.getConfidence() != null)
                //  recognition1ValueTextView.setText(
                //       String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
                //}

                // Recognition recognition2 = results.get(2);
                // if (recognition2 != null) {
                //  if (recognition2.getTitle() != null) recognition2TextView.setText(recognition2.getTitle());
                // if (recognition2.getConfidence() != null)
                //  recognition2ValueTextView.setText(
                //   String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
            }
        }
    }

    protected fun showFrameInfo(frameInfo: String) {
        frameValueTextView.setText(frameInfo)
    }

    protected fun showCropInfo(cropInfo: String) {
        cropValueTextView.setText(cropInfo)
    }

    protected fun showCameraResolution(cameraInfo: String) {
        cameraResolutionTextView.setText(previewWidth.toString() + "x" + previewHeight)
    }

    protected fun showRotationInfo(rotation: String) {
        rotationTextView.setText(rotation)
    }

    protected fun showInference(inferenceTime: String) {
        inferenceTimeTextView.setText(inferenceTime)
    }

    protected fun getModel(): Model {
        return model
    }

    private fun setModel(model: Model) {
        if (this.model !== model) {
            LOGGER.d("Updating  model: $model")
            this.model = model
            onInferenceConfigurationChanged()
        }
    }

    protected fun getDevice(): Device {
        return device
    }

    private fun setDevice(device: Device) {
        if (this.device !== device) {
            LOGGER.d("Updating  device: $device")
            this.device = device
            val threadsEnabled = device === Device.CPU
            plusImageView!!.setEnabled(threadsEnabled)
            minusImageView!!.setEnabled(threadsEnabled)
            threadsTextView!!.setText(if (threadsEnabled) String.valueOf(numThreads) else "N/A")
            onInferenceConfigurationChanged()
        }
    }

    protected fun getNumThreads(): Int {
        return numThreads
    }

    private fun setNumThreads(numThreads: Int) {
        if (this.numThreads != numThreads) {
            LOGGER.d("Updating  numThreads: $numThreads")
            this.numThreads = numThreads
            onInferenceConfigurationChanged()
        }
    }

    protected abstract fun processImage()

    protected abstract fun onPreviewSizeChosen(size: Size, rotation: Int)

    protected abstract fun onInferenceConfigurationChanged()

    @Override
    fun onClick(v: View) {
        if (v.getId() === R.id.plus) {
            val threads = threadsTextView!!.getText().toString().trim()
            var numThreads = Integer.parseInt(threads)
            if (numThreads >= 9) return
            setNumThreads(++numThreads)
            threadsTextView!!.setText(String.valueOf(numThreads))
        } else if (v.getId() === R.id.minus) {
            val threads = threadsTextView!!.getText().toString().trim()
            var numThreads = Integer.parseInt(threads)
            if (numThreads == 1) {
                return
            }
            setNumThreads(--numThreads)
            threadsTextView!!.setText(String.valueOf(numThreads))
        }
    }

    @Override
    fun onItemSelected(parent: AdapterView<*>, view: View, pos: Int, id: Long) {
        if (parent === modelSpinner) {
            setModel(Model.valueOf(parent.getItemAtPosition(pos).toString().toUpperCase()))
        } else if (parent === deviceSpinner) {
            setDevice(Device.valueOf(parent.getItemAtPosition(pos).toString()))
        }
    }

    @Override
    fun onNothingSelected(parent: AdapterView<*>) {
        // Do nothing.
    }

    companion object {
        private val LOGGER = Logger()

        private val PERMISSIONS_REQUEST = 1

        private val PERMISSION_CAMERA = Manifest.permission.CAMERA
    }
}
