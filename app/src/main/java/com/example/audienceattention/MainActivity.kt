package com.example.audienceattention

import android.Manifest
import android.graphics.Rect
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ConcurrentCamera
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.audienceattention.ui.theme.AudienceAttentionTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.*
import kotlinx.coroutines.delay
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.abs
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import android.util.Size

class MainActivity : ComponentActivity() {

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
        }
    @ExperimentalGetImage
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestPermissions.launch(
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            )
        )

        setContent {
            AudienceAttentionTheme {
                DualCameraSessionScreen()
            }
        }
    }
}


@ExperimentalGetImage
@Composable
fun DualCameraSessionScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // --- states ---
    val audienceAnalyzer = remember { AudienceAnalyzer() } // 너의 기존 Analyzer 사용

    var metrics by remember { mutableStateOf(CrowdMetrics.empty()) }
    var overlays by remember { mutableStateOf<List<FaceOverlay>>(emptyList()) }
    var statusText by remember { mutableStateOf("Initializing...") }
    var calibState by remember { mutableStateOf(audienceAnalyzer.getCalibrationState()) }

    // 로그 저장
    val logs = remember { mutableStateListOf<AttentionLogEntry>() }

    // 세션/녹화 상태
    var sessionRunning by remember { mutableStateOf(false) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var lastVideoUri by remember { mutableStateOf<Uri?>(null) }
    var lastJsonUri by remember { mutableStateOf<Uri?>(null) }
    var pendingShare by remember { mutableStateOf(false) }

    // 동시 카메라 가능 여부(녹화 bind 성공 시 true로 세팅)
    var frontRecordingAvailable by remember { mutableStateOf(false) }

    // PreviewView(후면 표시)
    val rearPreviewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } // 데모에서 bbox 정합이 더 쉬움
    }

    // 단일 executor
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // 전면 녹화 유스케이스
    val recorder = remember {
        Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.HD)) // 720p~HD 권장
            .build()
    }
    val videoCapture = remember { VideoCapture.withOutput(recorder) }

    // --- 1분마다 로그 기록: sessionRunning일 때만 ---
    LaunchedEffect(sessionRunning) {
        while (sessionRunning) {
            delay(60_000L)
            logs.add(
                AttentionLogEntry(
                    tsMs = System.currentTimeMillis(),
                    score1min = metrics.score1min,
                    faces = metrics.nFaces,
                    confidence = metrics.confidence
                )
            )
        }
    }

    // --- 카메라 bind: 한 번만 ---
    LaunchedEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()

        val rearPreview = Preview.Builder()
            .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_16_9)
            .build()
            .also { it.setSurfaceProvider(rearPreviewView.surfaceProvider) }

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            // Prefer 720p, but fall back safely if not supported
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                )
            )
            .build()

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//            .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_16_9)
            .setResolutionSelector(resolutionSelector)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            audienceAnalyzer.analyze(imageProxy) { m, ov ->
                metrics = m
                overlays = ov
                calibState = audienceAnalyzer.getCalibrationState()
                statusText = "Rear analyzing | faces=${m.nFaces} conf=${"%.2f".format(m.confidence)}"
            }
        }

        // front videoCapture는 이미 remember { VideoCapture.withOutput(recorder) } 로 생성되어 있다고 가정

        try {
            cameraProvider.unbindAll()

            // ====== 동시카메라 지원 확인 (front+back 세트 찾기) ======
            var frontSelector: CameraSelector? = null
            var backSelector: CameraSelector? = null

            for (pair in cameraProvider.availableConcurrentCameraInfos) {
                val hasFront = pair.any { it.lensFacing == CameraSelector.LENS_FACING_FRONT }
                val hasBack = pair.any { it.lensFacing == CameraSelector.LENS_FACING_BACK }
                if (hasFront && hasBack) {
                    frontSelector = pair.first { it.lensFacing == CameraSelector.LENS_FACING_FRONT }.cameraSelector
                    backSelector = pair.first { it.lensFacing == CameraSelector.LENS_FACING_BACK }.cameraSelector
                    break
                }
            }

            if (frontSelector == null || backSelector == null) {
                // ====== 동시카메라 미지원: 후면만 ======
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    rearPreview,
                    analysis
                )
                frontRecordingAvailable = false
                statusText = "Rear analyzing | Concurrent(front+back) not supported"
                return@LaunchedEffect
            }

            // ====== 동시카메라 지원: ConcurrentCamera로 bind ======
            val rearGroup = androidx.camera.core.UseCaseGroup.Builder()
                .addUseCase(rearPreview)
                .addUseCase(analysis)
                .build()

            val frontGroup = androidx.camera.core.UseCaseGroup.Builder()
                .addUseCase(videoCapture)
                .build()

            val rearConfig = ConcurrentCamera.SingleCameraConfig(
                backSelector!!,
                rearGroup,
                lifecycleOwner
            )

            val frontConfig = ConcurrentCamera.SingleCameraConfig(
                frontSelector!!,
                frontGroup,
                lifecycleOwner
            )

            cameraProvider.bindToLifecycle(listOf(rearConfig, frontConfig))

            frontRecordingAvailable = true
            statusText = "Rear analyzing + Front ready (Concurrent)"

        } catch (e: Exception) {
            frontRecordingAvailable = false
            statusText = "Camera bind failed: ${e.message}"
        }
    }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    // --- share pending when video finalize arrives ---
    LaunchedEffect(lastVideoUri, lastJsonUri, pendingShare) {
        if (pendingShare && lastVideoUri != null && lastJsonUri != null) {
            shareFiles(context, listOf(lastVideoUri!!, lastJsonUri!!))
            pendingShare = false
        }
    }

    /* ===================== UI ===================== */

    Box(modifier = Modifier.fillMaxSize()) {

        // Rear Preview
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { rearPreviewView }
        )

        // Face bbox overlay (데모용): FIT_CENTER 기준 근사 매핑
        Canvas(modifier = Modifier.fillMaxSize()) {
            val native = drawContext.canvas.nativeCanvas

            val stroke = android.graphics.Paint().apply {
                isAntiAlias = true
                style = android.graphics.Paint.Style.STROKE
                strokeWidth = 4f
                color = android.graphics.Color.GREEN
            }
            val textPaint = android.graphics.Paint().apply {
                isAntiAlias = true
                textSize = 34f
                color = android.graphics.Color.WHITE
            }
            val bgPaint = android.graphics.Paint().apply {
                isAntiAlias = true
                style = android.graphics.Paint.Style.FILL
                color = android.graphics.Color.argb(160, 0, 0, 0)
            }

            val vw = size.width
            val vh = size.height

            overlays.forEach { fo ->
                // overlays에 imgW/imgH가 있다고 가정 (없으면 너의 FaceOverlay에 추가해줘)
                val iw = fo.imgW
                val ih = fo.imgH
                if (iw <= 0 || ih <= 0) return@forEach

                // FIT_CENTER 매핑 (레터박스)
                val scale = min(vw / iw.toFloat(), vh / ih.toFloat())
                val dx = (vw - iw * scale) / 2f
                val dy = (vh - ih * scale) / 2f

                val r = android.graphics.RectF(fo.bbox)
                r.left = r.left * scale + dx
                r.right = r.right * scale + dx
                r.top = r.top * scale + dy
                r.bottom = r.bottom * scale + dy

                native.drawRect(r, stroke)

                val label = "ID ${fo.id}  ${fo.score100}"
                val tx = r.left
                val ty = max(40f, r.top - 10f)

                native.drawRect(
                    tx,
                    ty - 36f,
                    tx + textPaint.measureText(label) + 16f,
                    ty + 8f,
                    bgPaint
                )
                native.drawText(label, tx + 8f, ty, textPaint)
            }
        }

        // Top Overlay UI
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 16.dp)
        ) {
            Text(
                text = "Attention ${metrics.score1min} (1m)",
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = "faces=${metrics.nFaces}  conf=${"%.2f".format(metrics.confidence)}  logs=${logs.size}",
                style = MaterialTheme.typography.titleMedium
            )

            // Calibration status
            when (calibState.phase) {
                AttentionEstimatorWithCalibration.Phase.IDLE -> {
                    Text("Calibration: IDLE", style = MaterialTheme.typography.bodyMedium)
                }
                AttentionEstimatorWithCalibration.Phase.CALIBRATING -> {
                    val secLeft = ceil(calibState.remainingMs / 1000.0).toInt().coerceAtLeast(0)
                    Text("Calibration: CALIBRATING... ${secLeft}s", style = MaterialTheme.typography.bodyMedium)
                }
                AttentionEstimatorWithCalibration.Phase.RUNNING -> {
                    Text("Calibration: RUNNING", style = MaterialTheme.typography.bodyMedium)
                }
            }

            Text(text = statusText, style = MaterialTheme.typography.bodySmall)

            Spacer(modifier = Modifier.height(10.dp))

            // Session controls
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {

                Button(
                    enabled = !sessionRunning,
                    onClick = {
                        // start session = clear logs + start calibration + start recording (if available)
                        logs.clear()
                        lastJsonUri = null
                        lastVideoUri = null
                        pendingShare = false

                        audienceAnalyzer.startCalibration()
                        calibState = audienceAnalyzer.getCalibrationState()
                        sessionRunning = true

                        if (frontRecordingAvailable) {
                            recording = startFrontRecording(
                                context = context,
                                videoCapture = videoCapture,
                                onFinalize = { uri ->
                                    lastVideoUri = uri
                                }
                            )
                        }
                    }
                ) { Text(if (frontRecordingAvailable) "Start Session (Rec+Log)" else "Start Session (Log)") }

                Button(
                    enabled = sessionRunning,
                    onClick = {
                        sessionRunning = false

                        // stop recording if exists
                        recording?.stop()
                        recording = null

                        // save json now
                        lastJsonUri = writeLogsToJson(context, logs.toList())

                        // if video exists later, auto-share both
                        pendingShare = true

                        // if front recording wasn't available, share json only
                        if (!frontRecordingAvailable) {
                            shareFiles(context, listOf(lastJsonUri!!))
                            pendingShare = false
                        }
                    }
                ) { Text("Stop + Export") }
            }
        }
    }
}

private fun startFrontRecording(
    context: Context,
    videoCapture: VideoCapture<Recorder>,
    onFinalize: (Uri) -> Unit
): Recording {
    val name = "presenter_${System.currentTimeMillis()}.mp4"

    val values = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, name)
        put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
        put(MediaStore.MediaColumns.RELATIVE_PATH, "Movies/AudienceAttention")
    }

    val outputOptions = MediaStoreOutputOptions.Builder(
        context.contentResolver,
        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
    ).setContentValues(values).build()

    // NOTE: 오디오까지 원하면 .withAudioEnabled() 를 켜고 RECORD_AUDIO 권한 필요
    return videoCapture.output
        .prepareRecording(context, outputOptions)
        .withAudioEnabled()
        .start(ContextCompat.getMainExecutor(context)) { event ->
            when (event) {
                is VideoRecordEvent.Finalize -> {
                    if (!event.hasError()) {
                        onFinalize(event.outputResults.outputUri)
                    }
                }
            }
        }
}

/* =====================  Video + JSON I/O ===================== */
private fun writeLogsToJson(context: Context, logs: List<AttentionLogEntry>): Uri {
    val arr = org.json.JSONArray()
    logs.forEach { e ->
        arr.put(
            org.json.JSONObject().apply {
                put("tsMs", e.tsMs)
                put("score100_1min", e.score1min)
                put("faces", e.faces)
                put("confidence", e.confidence)
            }
        )
    }

    val jsonBytes = arr.toString(2).toByteArray()

    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
        // ===== API 29+ : MediaStore Downloads =====
        val name = "attention_${System.currentTimeMillis()}.json"
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/AudienceAttention")
        }

        val uri = context.contentResolver.insert(
            MediaStore.Downloads.EXTERNAL_CONTENT_URI,
            values
        ) ?: error("Failed to create JSON file")

        context.contentResolver.openOutputStream(uri)?.use { os ->
            os.write(jsonBytes)
        }
        uri

    } else {
        // ===== API 26~28 : app-specific external storage =====
        val dir = context.getExternalFilesDir(null)
            ?: error("External storage not available")

        val file = java.io.File(
            dir,
            "attention_${System.currentTimeMillis()}.json"
        )

        file.outputStream().use { it.write(jsonBytes) }

        // FileProvider 없이도 ACTION_SEND 가능 (같은 앱 내부 URI)
        androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
    }
}

private fun shareFiles(context: Context, uris: List<Uri>) {
    val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "*/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Export files"))
}

/* =====================  Vision + Scoring Core ===================== */
data class AttentionLogEntry(
    val tsMs: Long,
    val score1min: Int,
    val faces: Int,
    val confidence: Float
)

data class CrowdMetrics(
    val score1min: Int,
    val nFaces: Int,
    val confidence: Float,
) {
    companion object {
        fun empty() = CrowdMetrics(0, 0, 0f)
    }
}

data class TrackedFace(
    val id: Int,
    val face: Face,
    val lastSeenMs: Long
)

data class PersonState(
    val id: Int,
    val score: Float,     // EMA 0..1
    val quality: Float,   // 0..1
    val lastSeenMs: Long
)

data class FaceOverlay(
    val id: Int,
    val bbox: android.graphics.Rect,     // ML Kit boundingBox (image coords)
    val score100: Int,  // 0..100
    val quality: Float, // optional debug
    val imgW: Int,
    val imgH: Int
)
class AudienceAnalyzer {
    private val inFlight = AtomicBoolean(false)

    private val detector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .setMinFaceSize(0.02f)
            .build()
    )

    private val tracker = FaceTracker()
    private val estimator = AttentionEstimatorWithCalibration()
    private val aggregator = CrowdAggregator(windowSec1 = 60)

    fun startCalibration() {
        estimator.startCalibration(System.currentTimeMillis())
    }
    fun getCalibrationState(): AttentionEstimatorWithCalibration.CalibrationState {
        return estimator.getCalibrationState(System.currentTimeMillis())
    }

    @ExperimentalGetImage
    fun analyze(
        imageProxy: androidx.camera.core.ImageProxy,
        onResult: (CrowdMetrics, List<FaceOverlay>) -> Unit) {
        if (!inFlight.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            inFlight.set(false)
            imageProxy.close()
            return
        }
        val imgW = mediaImage.width
        val imgH = mediaImage.height

        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        val nowMs = System.currentTimeMillis()

        detector.process(inputImage)
            .addOnSuccessListener { faces ->
                val tracked = tracker.update(faces, nowMs)

                val people: List<PersonState> = tracked.mapNotNull { tf ->
                    val f = tf.face
                    estimator.update(
                        id = tf.id,
                        boundingBox = f.boundingBox,
                        headEulerAngleY = f.headEulerAngleY,
                        headEulerAngleX = f.headEulerAngleX,
                        leftEyeOpenProbability = f.leftEyeOpenProbability,
                        rightEyeOpenProbability = f.rightEyeOpenProbability,
                        nowMs = nowMs
                    )
                }

                val metrics = aggregator.update(people, nowMs)

                // id -> PersonState 매핑
                val byId = people.associateBy { it.id }

                // 얼굴별 overlay 만들기 (bbox + 점수)
                val overlays = tracked.mapNotNull { tf ->
                    val ps = byId[tf.id] ?: return@mapNotNull null
                    FaceOverlay(
                        id = tf.id,
                        bbox = tf.face.boundingBox,
                        score100 = (ps.score * 100f).toInt().coerceIn(0, 100),
                        quality = ps.quality,
                        imgW = imgW,
                        imgH = imgH
                    )
                }

                onResult(metrics, overlays)
            }

            .addOnFailureListener {
                // keep running
            }
            .addOnCompleteListener {
                inFlight.set(false)
                imageProxy.close()
            }
    }
}

/* =====================  Tracking ===================== */

class FaceTracker(
    private val ttlMs: Long = 1500L,
    private val iouThreshold: Float = 0.30f
) {
    private var nextId = 1
    private val tracks = mutableMapOf<Int, TrackedFace>()

    fun update(faces: List<Face>, nowMs: Long): List<TrackedFace> {
        // TTL 지난 트랙 제거
        val expired = tracks.values.filter { nowMs - it.lastSeenMs > ttlMs }.map { it.id }
        expired.forEach { tracks.remove(it) }

        val out = mutableListOf<TrackedFace>()

        for (f in faces) {
            val mlId = f.trackingId
            if (mlId != null) {
                val tf = TrackedFace(mlId, f, nowMs)
                tracks[mlId] = tf
                out.add(tf)
                continue
            }

            val (bestId, bestIou) = tracks.values
                .map { it.id to iou(it.face.boundingBox, f.boundingBox) }
                .maxByOrNull { it.second } ?: (null to 0f)

            val id = if (bestId != null && bestIou >= iouThreshold) bestId else nextId++
            val tf = TrackedFace(id, f, nowMs)
            tracks[id] = tf
            out.add(tf)
        }

        return out
    }

    private fun iou(a: Rect, b: Rect): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = max(0, interRight - interLeft)
        val interH = max(0, interBottom - interTop)
        val interArea = interW.toLong() * interH.toLong()
        val unionArea = a.width().toLong() * a.height().toLong() + b.width().toLong() * b.height().toLong() - interArea
        return if (unionArea <= 0L) 0f else interArea.toFloat() / unionArea.toFloat()
    }
}

/* =====================  Attention Scoring ===================== */

class AttentionEstimatorWithCalibration(
    private val calibrationDurationMs: Long = 10_000L,

    // Per-person EMA smoothing
    private val emaAlpha: Float = 0.10f,

    // Deviation thresholds (degrees) relative to baseline
    private val yawDevGood: Float = 15f,
    private val yawDevBad: Float = 40f,
    private val pitchDevGood: Float = 12f,
    private val pitchDevBad: Float = 35f,

    // Eye baseline drop thresholds
    private val eyeDropGood: Float = 0.10f,
    private val eyeDropBad: Float = 0.45f,

    // Face quality gating
    private val minQuality: Float = 0.15f,
    private val qualityAreaRef: Float = 40_000f,

    // Sustained behavior time thresholds (seconds)
    private val graceAwaySec: Float = 2.0f,
    private val fullAwaySec: Float = 8.0f,
    private val graceDownSec: Float = 1.5f,
    private val fullDownSec: Float = 6.0f,
    private val graceEyesSec: Float = 1.0f,
    private val fullEyesSec: Float = 4.5f,

    // Penalty strengths (0..1): max reduction amount at full penalty
    private val awayPenaltyStrength: Float = 0.55f,
    private val downPenaltyStrength: Float = 0.65f,
    private val eyesPenaltyStrength: Float = 0.70f
) {

    enum class Phase { IDLE, CALIBRATING, RUNNING }

    data class CalibrationState(
        val phase: Phase,
        val startedAtMs: Long?,
        val elapsedMs: Long,
        val remainingMs: Long,
        val yawCenter: Float?,
        val pitchCenter: Float?,
        val eyeBaseline: Float?
    )

    private var phase: Phase = Phase.IDLE
    private var calibrationStartMs: Long? = null

    // Learned baseline (global)
    private var yawCenter: Float? = null
    private var pitchCenter: Float? = null
    private var eyeBaseline: Float? = null

    // Calibration samples (quality filtered)
    private val yawSamples = ArrayList<Float>(1024)
    private val pitchSamples = ArrayList<Float>(1024)
    private val eyeSamples = ArrayList<Float>(1024)

    // Per-person EMA state: store last EMA score + last seen
    private data class InternalState(val ema: Float, val lastSeenMs: Long)
    private val internal = mutableMapOf<Int, InternalState>()

    // Per-person accumulators (seconds)
    private val awaySec = mutableMapOf<Int, Float>()
    private val downSec = mutableMapOf<Int, Float>()
    private val eyesLowSec = mutableMapOf<Int, Float>()

    fun startCalibration(nowMs: Long = System.currentTimeMillis()) {
        phase = Phase.CALIBRATING
        calibrationStartMs = nowMs

        yawCenter = null
        pitchCenter = null
        eyeBaseline = null

        yawSamples.clear()
        pitchSamples.clear()
        eyeSamples.clear()

        // Reset accumulators for clean session behavior
        awaySec.clear()
        downSec.clear()
        eyesLowSec.clear()
    }

    fun getCalibrationState(nowMs: Long = System.currentTimeMillis()): CalibrationState {
        val start = calibrationStartMs
        val elapsed = if (start == null) 0L else max(0L, nowMs - start)
        val remaining = if (phase == Phase.CALIBRATING) max(0L, calibrationDurationMs - elapsed) else 0L
        return CalibrationState(
            phase = phase,
            startedAtMs = start,
            elapsedMs = elapsed,
            remainingMs = remaining,
            yawCenter = yawCenter,
            pitchCenter = pitchCenter,
            eyeBaseline = eyeBaseline
        )
    }

    /**
     * Main update.
     * Returns TOP-LEVEL PersonState (id, score, quality, lastSeenMs) or null if face quality too low.
     */
    fun update(
        id: Int,
        boundingBox: Rect,
        headEulerAngleY: Float,
        headEulerAngleX: Float,
        leftEyeOpenProbability: Float?,
        rightEyeOpenProbability: Float?,
        nowMs: Long = System.currentTimeMillis()
    ): PersonState? {

        val q = faceQuality(boundingBox)
        if (q < minQuality) return null

        val yawSigned = headEulerAngleY
        val pitchSigned = headEulerAngleX
        val eyeOpen = computeEyeOpen(leftEyeOpenProbability, rightEyeOpenProbability)

        // Calibration mode: collect samples and output a "soft" score (so UI doesn't freeze)
        if (phase == Phase.CALIBRATING) {
            if (q >= 0.35f) {
                yawSamples.add(yawSigned)
                pitchSamples.add(pitchSigned)
                eyeSamples.add(eyeOpen)
            }

            val start = calibrationStartMs ?: nowMs
            if (nowMs - start >= calibrationDurationMs) {
                finishCalibration(nowMs)
            }

            val rawDuringCalib = (0.60f * 1.0f + 0.40f * eyeOpen)
            val ema = updateEma(id, rawDuringCalib, nowMs)
            return PersonState(id, ema, q, nowMs)
        }

        // If not calibrated yet, fall back to absolute camera-facing scoring
        if (phase == Phase.IDLE || yawCenter == null || pitchCenter == null || eyeBaseline == null) {
            val rawFallback = rawAbsoluteCameraFacing(
                yawSigned = yawSigned,
                pitchSigned = pitchSigned,
                eyeOpen = eyeOpen,
                q = q
            )
            val ema = updateEma(id, rawFallback, nowMs)
            return PersonState(id, ema, q, nowMs)
        }

        // RUNNING: baseline deviation scoring
        val yc = yawCenter!!
        val pc = pitchCenter!!
        val eb = eyeBaseline!!

        val yawDev = abs(yawSigned - yc)
        val pitchDev = abs(pitchSigned - pc)
        val eyeDrop = max(0f, eb - eyeOpen)

        val sYawDev = scoreBetween(yawDev, yawDevGood, yawDevBad)
        val sPitchDev = scoreBetween(pitchDev, pitchDevGood, pitchDevBad)
        val sPose = 0.70f * sYawDev + 0.30f * sPitchDev

        // eye score: if eyeDrop small => good; if large => bad
        val sEye = 1f - scoreBetween(eyeDrop, eyeDropGood, eyeDropBad)

        // dt seconds for sustained penalties
        val prev = internal[id]
        val dtSec = if (prev == null) (1f / 15f) else ((nowMs - prev.lastSeenMs).coerceAtMost(250L) / 1000f)

        // sustained conditions (tuneable)
        val isAway = yawDev > 30f
        val isDown = (pitchDev > 25f) || (pitchSigned > (pc + 20f)) // more downward than baseline
        val isEyesLow = eyeOpen < max(0.15f, eb - 0.30f)

        awaySec[id] = updateAccumulator(awaySec[id] ?: 0f, isAway, dtSec, decayRate = 2.0f)
        downSec[id] = updateAccumulator(downSec[id] ?: 0f, isDown, dtSec, decayRate = 2.5f)
        eyesLowSec[id] = updateAccumulator(eyesLowSec[id] ?: 0f, isEyesLow, dtSec, decayRate = 3.0f)

        val awayFactor = 1f - awayPenaltyStrength * timePenalty(awaySec[id]!!, graceAwaySec, fullAwaySec)
        val downFactor = 1f - downPenaltyStrength * timePenalty(downSec[id]!!, graceDownSec, fullDownSec)
        val eyesFactor = 1f - eyesPenaltyStrength * timePenalty(eyesLowSec[id]!!, graceEyesSec, fullEyesSec)

        val wEye = (0.15f + 0.35f * q).coerceIn(0.15f, 0.50f)  // 멀면 eye 비중 낮춤
        val wPose = 1f - wEye
        val base = wPose * sPose + wEye * sEye
        // val base = 0.65f * sPose + 0.35f * sEye
        val raw = (base * awayFactor * downFactor * eyesFactor).coerceIn(0f, 1f)

        val ema = updateEma(id, raw, nowMs)
        return PersonState(id, ema, q, nowMs)
    }

    // ===================== internal helpers =====================

    private fun finishCalibration(nowMs: Long) {
        yawCenter = medianOrNull(yawSamples) ?: 0f
        pitchCenter = medianOrNull(pitchSamples) ?: 0f
        eyeBaseline = (meanOrNull(eyeSamples) ?: 0.70f).coerceIn(0f, 1f)

        phase = Phase.RUNNING
        // Keep EMA states; they will adapt naturally.
        // Optionally reset internal EMA here if you prefer a fresh start after calib.
        // internal.clear()
    }

    private fun updateEma(id: Int, raw: Float, nowMs: Long): Float {
        val prev = internal[id]
        val ema = if (prev == null) raw else ((1 - emaAlpha) * prev.ema + emaAlpha * raw)
        internal[id] = InternalState(ema = ema.coerceIn(0f, 1f), lastSeenMs = nowMs)
        return internal[id]!!.ema
    }

    private fun faceQuality(b: Rect): Float {
        val area = (b.width() * b.height()).toFloat()
        return (area / qualityAreaRef).coerceIn(0f, 1f)
    }

    private fun computeEyeOpen(le: Float?, re: Float?): Float {
        val v = when {
            le != null && re != null -> (le + re) / 2f
            le != null -> le
            re != null -> re
            else -> 0.70f
        }
        return v.coerceIn(0f, 1f)
    }

    private fun scoreBetween(v: Float, good: Float, bad: Float): Float {
        // v <= good -> 1, v >= bad -> 0
        return when {
            v <= good -> 1f
            v >= bad -> 0f
            else -> 1f - (v - good) / (bad - good)
        }
    }

    private fun timePenalty(sec: Float, grace: Float, full: Float): Float {
        if (sec <= grace) return 0f
        if (sec >= full) return 1f
        return ((sec - grace) / (full - grace)).coerceIn(0f, 1f)
    }

    private fun updateAccumulator(current: Float, active: Boolean, dtSec: Float, decayRate: Float): Float {
        val next = if (active) current + dtSec else current - decayRate * dtSec
        return next.coerceIn(0f, 60f)
    }

    private fun medianOrNull(list: List<Float>): Float? {
        if (list.isEmpty()) return null
        val sorted = list.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 1) {
            sorted[mid]
        } else {
            (sorted[mid - 1] + sorted[mid]) / 2f
        }
    }

    private fun meanOrNull(list: List<Float>): Float? {
        if (list.isEmpty()) return null
        var s = 0.0
        for (v in list) s += v.toDouble()
        return (s / list.size).toFloat()
    }

    private fun rawAbsoluteCameraFacing(
        yawSigned: Float,
        pitchSigned: Float,
        eyeOpen: Float,
        q: Float
    ): Float {
        val yaw = abs(yawSigned)
        val pitch = abs(pitchSigned)
        val sYaw = scoreBetween(yaw, good = 30f, bad = 55f)
        val sPitch = scoreBetween(pitch, good = 25f, bad = 45f)
        val sPose = 0.70f * sYaw + 0.30f * sPitch
        return (0.65f * sPose + 0.35f * eyeOpen).coerceIn(0f, 1f)
    }
}

/* =====================  Aggregation (1m) ===================== */

class CrowdAggregator(
    private val windowSec1: Int = 60
) {
    private val buf1 = ArrayDeque<Pair<Long, Float>>()

    fun update(people: List<PersonState>, nowMs: Long): CrowdMetrics {
        val n = people.size
        val meanScore = if (n == 0) 0f else (people.sumOf { it.score.toDouble() }.toFloat() / n)
        val meanQ = if (n == 0) 0f else (people.sumOf { it.quality.toDouble() }.toFloat() / n)

        push(buf1, nowMs, meanScore, windowSec1)
        val a1 = avg(buf1)
        val conf = confidence(n, meanQ)

        return CrowdMetrics(
            score1min = (a1 * 100f).toInt().coerceIn(0, 100),
            nFaces = n,
            confidence = conf
        )
    }

    private fun push(buf: ArrayDeque<Pair<Long, Float>>, nowMs: Long, value: Float, windowSec: Int) {
        buf.addLast(nowMs to value)
        val cutoff = nowMs - windowSec * 1000L
        while (buf.isNotEmpty() && buf.first().first < cutoff) buf.removeFirst()
    }

    private fun avg(buf: ArrayDeque<Pair<Long, Float>>): Float {
        if (buf.isEmpty()) return 0f
        return (buf.sumOf { it.second.toDouble() }.toFloat() / buf.size)
    }

    private fun confidence(nFaces: Int, meanQ: Float): Float {
        val nFactor = (nFaces / 8f).coerceIn(0f, 1f)
        return (0.6f * nFactor + 0.4f * meanQ).coerceIn(0f, 1f)
    }
}
