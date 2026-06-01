import json
import logging
import re
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QMessageBox,
    QSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from model_manager import download_asr, download_silero
from i18n import t

log = logging.getLogger("LiveTranslate.Dialogs")

SETTINGS_FILE = None  # set by control_panel on import
_save_settings = None  # set by control_panel on import


class _LogCapture(logging.Handler):
    """Captures log output and emits via callback."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self._callback(self.format(record))
        except Exception:
            pass


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _StderrCapture:
    """Captures stderr (tqdm) and forwards cleaned lines via callback."""

    def __init__(self, callback, original):
        self._cb = callback
        self._orig = original

    def write(self, text):
        if self._orig:
            self._orig.write(text)
        if not text:
            return
        cleaned = _ANSI_RE.sub("", text)
        for line in cleaned.splitlines():
            line = line.strip()
            if line:
                self._cb(line)

    def flush(self):
        if self._orig:
            self._orig.flush()

    def isatty(self):
        return False


class _ModelLoadDialog(QDialog):
    """Modal dialog shown during model download/loading with live log."""

    _log_signal = pyqtSignal(str)

    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LiveTranslate")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout(self)
        self._label = QLabel(message)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 8))
        self._log_view.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        layout.addWidget(self._log_view)

        self._log_signal.connect(self._append_log)
        self._log_handler = _LogCapture(self._log_signal.emit)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._log_handler)

    def _append_log(self, text):
        self._log_view.append(text)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    def done(self, result):
        logging.getLogger().removeHandler(self._log_handler)
        super().done(result)


class SetupWizardDialog(QDialog):
    """First-launch wizard: choose hub, download models."""

    _log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("window_setup"))
        self.setMinimumWidth(520)
        self.setMinimumHeight(400)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout(self)

        hub_group = QGroupBox(t("group_download_source"))
        hub_layout = QVBoxLayout(hub_group)
        self._hub_combo = QComboBox()
        self._hub_combo.addItems(
            [
                t("hub_modelscope_full"),
                t("hub_huggingface_full"),
            ]
        )
        hub_layout.addWidget(self._hub_combo)
        layout.addWidget(hub_group)

        self._download_btn = QPushButton(t("btn_start_download"))
        self._download_btn.clicked.connect(self._start_download)
        layout.addWidget(self._download_btn)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 8))
        self._log_view.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        self._log_view.hide()
        layout.addWidget(self._log_view)

        self._error = None
        self._log_signal.connect(self._append_log)
        self._log_handler = _LogCapture(self._log_signal.emit)

        # Auto-start countdown
        self._countdown = 5
        self._auto_timer = QTimer()
        self._auto_timer.setInterval(1000)
        self._auto_timer.timeout.connect(self._tick_countdown)
        self._auto_timer.start()
        self._update_btn_countdown()

        self._hub_combo.currentIndexChanged.connect(self._reset_countdown)

    def _update_btn_countdown(self):
        self._download_btn.setText(
            f"{t('btn_start_download')} ({self._countdown}s)"
        )

    def _reset_countdown(self):
        self._countdown = 5
        self._auto_timer.start()
        self._update_btn_countdown()

    def _tick_countdown(self):
        self._countdown -= 1
        if self._countdown <= 0:
            self._auto_timer.stop()
            self._start_download()
        else:
            self._update_btn_countdown()

    def _append_log(self, text):
        self._log_view.append(text)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    def _start_download(self):
        self._auto_timer.stop()
        self._download_btn.setText(t("btn_start_download"))
        self._download_btn.setEnabled(False)
        self._hub_combo.setEnabled(False)
        self._log_view.show()

        hub = "ms" if self._hub_combo.currentIndex() == 0 else "hf"

        logging.getLogger().addHandler(self._log_handler)
        self._orig_stderr = sys.stderr
        sys.stderr = _StderrCapture(self._log_signal.emit, self._orig_stderr)

        self._error = None
        self._download_thread = threading.Thread(
            target=self._download_worker, args=(hub,), daemon=True
        )
        self._download_thread.start()

        self._poll_timer = QTimer()
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._check_done)
        self._poll_timer.start()

    def _download_worker(self, hub):
        try:
            download_silero()
            download_asr("sensevoice", hub=hub)
        except Exception as e:
            self._error = str(e)
            log.error(f"Download failed: {e}", exc_info=True)

    def _check_done(self):
        if self._download_thread.is_alive():
            return
        self._poll_timer.stop()
        sys.stderr = self._orig_stderr
        logging.getLogger().removeHandler(self._log_handler)

        if self._error:
            self._append_log(f"\n{t('download_failed').format(error=self._error)}")
            self._download_btn.setEnabled(True)
            self._download_btn.setText(t("btn_retry"))
            self._hub_combo.setEnabled(True)
            return

        self._append_log(f"\n{t('download_complete')}")
        hub = "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        from control_panel import _save_settings

        settings = {
            "hub": hub,
            "asr_engine": "sensevoice",
            "vad_mode": "silero",
            "vad_threshold": 0.3,
            "energy_threshold": 0.02,
            "min_speech_duration": 1.0,
            "max_speech_duration": 8.0,
            "silence_mode": "auto",
            "silence_duration": 0.8,
            "asr_language": "auto",
            "target_language": "zh",
        }
        _save_settings(settings)
        QTimer.singleShot(500, self.accept)


class ModelDownloadDialog(QDialog):
    """Download missing models (non-first-launch) with live log."""

    _log_signal = pyqtSignal(str)

    def __init__(self, missing_models, hub="ms", parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("window_download"))
        self.setMinimumWidth(520)
        self.setMinimumHeight(300)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout(self)

        names = ", ".join(m["name"] for m in missing_models)
        info = QLabel(t("downloading_models").format(names=names))
        info.setWordWrap(True)
        layout.addWidget(info)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 8))
        self._log_view.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        layout.addWidget(self._log_view)

        self._close_btn = QPushButton(t("btn_close"))
        self._close_btn.clicked.connect(self.reject)
        self._close_btn.hide()
        layout.addWidget(self._close_btn)

        self._missing = missing_models
        self._hub = hub
        self._error = None

        self._log_signal.connect(self._append_log)
        self._log_handler = _LogCapture(self._log_signal.emit)

        QTimer.singleShot(100, self._start_download)

    def _append_log(self, text):
        self._log_view.append(text)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    def _start_download(self):
        logging.getLogger().addHandler(self._log_handler)
        self._orig_stderr = sys.stderr
        sys.stderr = _StderrCapture(self._log_signal.emit, self._orig_stderr)

        self._download_thread = threading.Thread(
            target=self._download_worker, daemon=True
        )
        self._download_thread.start()

        self._poll_timer = QTimer()
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._check_done)
        self._poll_timer.start()

    def _download_worker(self):
        try:
            for m in self._missing:
                if m["type"] == "silero-vad":
                    download_silero()
                elif m["type"] in (
                    "sensevoice",
                    "funasr-nano",
                    "funasr-mlt-nano",
                    "anime-whisper",
                ):
                    download_asr(m["type"], hub=self._hub)
                elif m["type"].startswith("whisper-"):
                    size = m["type"].replace("whisper-", "")
                    download_asr("whisper", model_size=size, hub=self._hub)
        except Exception as e:
            self._error = str(e)
            log.error(f"Download failed: {e}", exc_info=True)

    def _check_done(self):
        if self._download_thread.is_alive():
            return
        self._poll_timer.stop()
        sys.stderr = self._orig_stderr
        logging.getLogger().removeHandler(self._log_handler)

        if self._error:
            self._append_log(f"\n{t('download_failed').format(error=self._error)}")
            self._close_btn.show()
            return

        self._append_log(f"\n{t('download_complete')}")
        QTimer.singleShot(500, self.accept)


class ModelEditDialog(QDialog):
    """Dialog for adding/editing a model configuration.

    Supports both LLM (OpenAI-compatible) and traditional translation APIs
    (Baidu, Tencent, Youdao, DeepL).  The ``type`` field in model_data
    determines which UI is shown (default ``"llm"`` for backward compat).
    """

    def __init__(self, parent=None, model_data=None):
        super().__init__(parent)
        self.setWindowTitle(
            t("dialog_edit_model") if model_data else t("dialog_add_model")
        )
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)

        # ---- Translator type selector ----
        type_group = QGroupBox()
        type_group.setFlat(True)
        type_form = QFormLayout(type_group)
        self._tl_type = QComboBox()
        self._tl_type.addItem(t("translator_llm"), "llm")
        self._tl_type.addItem(t("translator_baidu"), "baidu")
        self._tl_type.addItem(t("translator_tencent"), "tencent")
        self._tl_type.addItem(t("translator_youdao"), "youdao")
        self._tl_type.addItem(t("translator_deepl"), "deepl")
        self._tl_type.currentIndexChanged.connect(self._on_type_changed)
        type_form.addRow(t("label_translator_type"), self._tl_type)
        root.addWidget(type_group)

        # ---- Common fields (name + proxy) ----
        common_group = QGroupBox()
        common_group.setFlat(True)
        common_form = QFormLayout(common_group)
        self._name = QLineEdit()
        self._proxy_mode = QComboBox()
        self._proxy_mode.addItems(
            [t("proxy_none"), t("proxy_system"), t("proxy_custom")]
        )
        self._proxy_mode.currentIndexChanged.connect(self._on_proxy_mode_changed)
        self._proxy_url = QLineEdit()
        self._proxy_url.setPlaceholderText("http://127.0.0.1:7890")
        self._proxy_url.setEnabled(False)
        common_form.addRow(t("label_display_name"), self._name)
        common_form.addRow(t("label_proxy"), self._proxy_mode)
        common_form.addRow(t("label_proxy_url"), self._proxy_url)
        root.addWidget(common_group)

        # ---- Stacked widget: LLM page vs Traditional API page ----
        self._stack = QStackedWidget()

        # == Page 0: LLM config ==
        llm_page = QWidget()
        llm_form = QFormLayout(llm_page)

        self._api_base = QLineEdit()
        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._model = QLineEdit()

        self._no_system_role = QCheckBox(t("no_system_role"))
        self._no_system_role.setToolTip(t("no_system_role_hint"))
        self._no_think = QCheckBox(t("no_think"))
        self._no_think.setToolTip(t("no_think_hint"))
        self._no_think.setChecked(True)
        self._streaming = QCheckBox(t("streaming"))
        self._streaming.setToolTip(t("streaming_hint"))
        self._streaming.setChecked(True)
        self._json_response = QCheckBox(t("json_response"))
        self._json_response.setToolTip(t("json_response_hint"))
        self._context_turns = QSpinBox()
        self._context_turns.setRange(0, 20)
        self._context_turns.setValue(0)
        self._context_turns.setToolTip(t("context_turns_hint"))

        price_suffix = t("price_suffix")
        self._input_price = QDoubleSpinBox()
        self._input_price.setRange(0, 999)
        self._input_price.setDecimals(2)
        self._input_price.setSuffix(price_suffix)
        self._input_price.setSpecialValueText("—")
        self._output_price = QDoubleSpinBox()
        self._output_price.setRange(0, 999)
        self._output_price.setDecimals(2)
        self._output_price.setSuffix(price_suffix)
        self._output_price.setSpecialValueText("—")

        price_row = QHBoxLayout()
        price_row.addWidget(QLabel(t("label_input_price")))
        price_row.addWidget(self._input_price)
        price_row.addWidget(QLabel(t("label_output_price")))
        price_row.addWidget(self._output_price)

        llm_form.addRow(t("label_api_base"), self._api_base)
        llm_form.addRow(t("label_api_key"), self._api_key)
        llm_form.addRow(t("label_model"), self._model)
        llm_form.addRow(t("label_pricing"), price_row)
        llm_form.addRow(t("label_context_turns"), self._context_turns)
        llm_form.addRow("", self._streaming)
        llm_form.addRow("", self._json_response)
        llm_form.addRow("", self._no_system_role)
        llm_form.addRow("", self._no_think)

        self._stack.addWidget(llm_page)  # index 0

        # == Page 1: Traditional API config ==
        api_page = QWidget()
        api_layout = QVBoxLayout(api_page)
        api_layout.setContentsMargins(0, 0, 0, 0)

        # Sub-stacked widget for API-specific fields
        self._api_stack = QStackedWidget()

        # -- Baidu --
        baidu_page = QWidget()
        baidu_form = QFormLayout(baidu_page)
        self._baidu_app_id = QLineEdit()
        self._baidu_secret_key = QLineEdit()
        self._baidu_secret_key.setEchoMode(QLineEdit.EchoMode.Password)
        baidu_form.addRow(t("label_app_id"), self._baidu_app_id)
        baidu_form.addRow(t("label_secret_key"), self._baidu_secret_key)
        self._api_stack.addWidget(baidu_page)  # index 0

        # -- Tencent --
        tencent_page = QWidget()
        tencent_form = QFormLayout(tencent_page)
        self._tencent_secret_id = QLineEdit()
        self._tencent_secret_key = QLineEdit()
        self._tencent_secret_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._tencent_region = QComboBox()
        regions = [
            ("ap-guangzhou", "ap-guangzhou"),
            ("ap-beijing", "ap-beijing"),
            ("ap-shanghai", "ap-shanghai"),
            ("ap-hongkong", "ap-hongkong"),
            ("na-siliconvalley", "na-siliconvalley"),
        ]
        for label, data in regions:
            self._tencent_region.addItem(label, data)
        tencent_form.addRow(t("label_secret_id"), self._tencent_secret_id)
        tencent_form.addRow(t("label_secret_key"), self._tencent_secret_key)
        tencent_form.addRow(t("label_region"), self._tencent_region)
        self._api_stack.addWidget(tencent_page)  # index 1

        # -- Youdao --
        youdao_page = QWidget()
        youdao_form = QFormLayout(youdao_page)
        self._youdao_app_key = QLineEdit()
        self._youdao_app_secret = QLineEdit()
        self._youdao_app_secret.setEchoMode(QLineEdit.EchoMode.Password)
        youdao_form.addRow(t("label_app_key"), self._youdao_app_key)
        youdao_form.addRow(t("label_app_secret"), self._youdao_app_secret)
        self._api_stack.addWidget(youdao_page)  # index 2

        # -- DeepL --
        deepl_page = QWidget()
        deepl_form = QFormLayout(deepl_page)
        self._deepl_api_key = QLineEdit()
        self._deepl_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        deepl_form.addRow(t("label_api_key"), self._deepl_api_key)
        self._api_stack.addWidget(deepl_page)  # index 3

        api_layout.addWidget(self._api_stack)
        self._stack.addWidget(api_page)  # index 1

        root.addWidget(self._stack)

        # ---- LLM-only: Advanced section ----
        self._adv_group = QGroupBox(t("label_advanced_params"))
        adv_layout = QFormLayout(self._adv_group)
        self._adv_group.setToolTip(t("override_hint"))

        self._adv_temperature = QDoubleSpinBox()
        self._adv_temperature.setRange(0.0, 2.0)
        self._adv_temperature.setDecimals(2)
        self._adv_temperature.setSingleStep(0.1)
        self._adv_temperature.setValue(0.3)

        self._adv_top_p = QDoubleSpinBox()
        self._adv_top_p.setRange(0.0, 1.0)
        self._adv_top_p.setDecimals(2)
        self._adv_top_p.setSingleStep(0.05)
        self._adv_top_p.setValue(1.0)

        self._adv_max_tokens = QSpinBox()
        self._adv_max_tokens.setRange(1, 32768)
        self._adv_max_tokens.setValue(256)

        self._adv_freq_penalty = QDoubleSpinBox()
        self._adv_freq_penalty.setRange(-2.0, 2.0)
        self._adv_freq_penalty.setDecimals(2)
        self._adv_freq_penalty.setSingleStep(0.1)

        self._adv_presence_penalty = QDoubleSpinBox()
        self._adv_presence_penalty.setRange(-2.0, 2.0)
        self._adv_presence_penalty.setDecimals(2)
        self._adv_presence_penalty.setSingleStep(0.1)

        self._adv_seed = QSpinBox()
        self._adv_seed.setRange(0, 2_000_000_000)

        self._adv_rows = {
            "temperature": self._make_override_row(self._adv_temperature),
            "top_p": self._make_override_row(self._adv_top_p),
            "max_tokens": self._make_override_row(self._adv_max_tokens),
            "frequency_penalty": self._make_override_row(self._adv_freq_penalty),
            "presence_penalty": self._make_override_row(self._adv_presence_penalty),
            "seed": self._make_override_row(self._adv_seed),
        }
        adv_layout.addRow(t("label_temperature"), self._adv_rows["temperature"][1])
        adv_layout.addRow(t("label_top_p"), self._adv_rows["top_p"][1])
        adv_layout.addRow(t("label_max_tokens"), self._adv_rows["max_tokens"][1])
        adv_layout.addRow(
            t("label_frequency_penalty"), self._adv_rows["frequency_penalty"][1]
        )
        adv_layout.addRow(
            t("label_presence_penalty"), self._adv_rows["presence_penalty"][1]
        )
        adv_layout.addRow(t("label_seed"), self._adv_rows["seed"][1])

        self._adv_extra_body = QTextEdit()
        self._adv_extra_body.setPlaceholderText(
            '{"thinking_budget": 1024}'
        )
        self._adv_extra_body.setToolTip(t("extra_body_hint"))
        self._adv_extra_body.setFixedHeight(70)
        adv_layout.addRow(t("label_extra_body"), self._adv_extra_body)

        root.addWidget(self._adv_group)

        # ---- Populate from model_data ----
        if model_data:
            self._name.setText(model_data.get("name", ""))
            proxy = model_data.get("proxy", "none")
            if proxy == "system":
                self._proxy_mode.setCurrentIndex(1)
            elif proxy not in ("none", "system") and proxy:
                self._proxy_mode.setCurrentIndex(2)
                self._proxy_url.setText(proxy)
            else:
                self._proxy_mode.setCurrentIndex(0)

            # Type selector
            tl_type = model_data.get("type", "llm")
            idx = self._tl_type.findData(tl_type)
            if idx >= 0:
                self._tl_type.setCurrentIndex(idx)

            # LLM fields
            self._api_base.setText(model_data.get("api_base", ""))
            self._api_key.setText(model_data.get("api_key", ""))
            self._model.setText(model_data.get("model", ""))
            self._no_system_role.setChecked(model_data.get("no_system_role", False))
            self._no_think.setChecked(model_data.get("no_think", True))
            self._streaming.setChecked(model_data.get("streaming", True))
            self._json_response.setChecked(model_data.get("json_response", False))
            self._context_turns.setValue(model_data.get("context_turns", 0))
            self._input_price.setValue(model_data.get("input_price", 0))
            self._output_price.setValue(model_data.get("output_price", 0))

            overrides = model_data.get("overrides") or {}
            for key, (cb, _row, widget) in self._adv_rows.items():
                if key in overrides and overrides[key] is not None:
                    cb.setChecked(True)
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(overrides[key]))
                    else:
                        widget.setValue(float(overrides[key]))
            extra_body = model_data.get("extra_body")
            if extra_body:
                try:
                    self._adv_extra_body.setPlainText(
                        json.dumps(extra_body, ensure_ascii=False, indent=2)
                    )
                except (TypeError, ValueError):
                    pass

            # Traditional API fields
            self._baidu_app_id.setText(model_data.get("app_id", ""))
            self._baidu_secret_key.setText(model_data.get("secret_key", ""))
            self._tencent_secret_id.setText(model_data.get("secret_id", ""))
            self._tencent_secret_key.setText(model_data.get("secret_key", ""))
            region = model_data.get("region", "ap-guangzhou")
            ridx = self._tencent_region.findData(region)
            if ridx >= 0:
                self._tencent_region.setCurrentIndex(ridx)
            self._youdao_app_key.setText(model_data.get("app_key", ""))
            self._youdao_app_secret.setText(model_data.get("app_secret", ""))
            self._deepl_api_key.setText(model_data.get("api_key", ""))

        # Trigger initial visibility
        self._on_type_changed()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---- helpers ----

    def _make_override_row(self, widget):
        """Build a [checkbox + widget] row that disables the widget when unchecked."""
        cb = QCheckBox(t("override_enable"))
        widget.setEnabled(False)
        cb.toggled.connect(widget.setEnabled)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(cb)
        h.addWidget(widget, 1)
        return cb, row, widget

    def _on_proxy_mode_changed(self, index):
        self._proxy_url.setEnabled(index == 2)

    def _on_type_changed(self):
        """Show/hide LLM vs traditional API fields based on selected type."""
        tl_type = self._tl_type.currentData()
        is_llm = tl_type == "llm"
        self._stack.setCurrentIndex(0 if is_llm else 1)
        self._adv_group.setVisible(is_llm)

        # Switch API-specific page inside the traditional API stack
        if not is_llm:
            api_index = {"baidu": 0, "tencent": 1, "youdao": 2, "deepl": 3}
            self._api_stack.setCurrentIndex(api_index.get(tl_type, 0))

    def _parse_extra_body(self):
        """Return (ok, data_or_error_msg). Empty text → (True, None)."""
        text = self._adv_extra_body.toPlainText().strip()
        if not text:
            return True, None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return False, f"{e}"
        if not isinstance(data, dict):
            return False, "extra_body must be a JSON object"
        return True, data

    def _on_accept(self):
        tl_type = self._tl_type.currentData()
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, t("error_title"), "Name is required.")
            return

        if tl_type == "llm":
            ok, _ = self._parse_extra_body()
            if not ok:
                QMessageBox.warning(
                    self, t("error_title"), t("extra_body_invalid")
                )
                return
        elif tl_type == "baidu":
            if not self._baidu_app_id.text().strip() or not self._baidu_secret_key.text().strip():
                QMessageBox.warning(self, t("error_title"), "App ID and Secret Key are required.")
                return
        elif tl_type == "tencent":
            if not self._tencent_secret_id.text().strip() or not self._tencent_secret_key.text().strip():
                QMessageBox.warning(self, t("error_title"), "Secret ID and Secret Key are required.")
                return
        elif tl_type == "youdao":
            if not self._youdao_app_key.text().strip() or not self._youdao_app_secret.text().strip():
                QMessageBox.warning(self, t("error_title"), "App Key and App Secret are required.")
                return
        elif tl_type == "deepl":
            if not self._deepl_api_key.text().strip():
                QMessageBox.warning(self, t("error_title"), "API Key is required.")
                return
        self.accept()

    def _proxy_value(self) -> str:
        proxy_idx = self._proxy_mode.currentIndex()
        if proxy_idx == 1:
            return "system"
        elif proxy_idx == 2:
            return self._proxy_url.text().strip() or "none"
        return "none"

    def get_data(self) -> dict:
        tl_type = self._tl_type.currentData()
        result = {
            "name": self._name.text().strip(),
            "type": tl_type,
            "proxy": self._proxy_value(),
        }

        if tl_type == "llm":
            result["api_base"] = self._api_base.text().strip()
            result["api_key"] = self._api_key.text().strip()
            result["model"] = self._model.text().strip()
            if self._no_system_role.isChecked():
                result["no_system_role"] = True
            if not self._no_think.isChecked():
                result["no_think"] = False
            if not self._streaming.isChecked():
                result["streaming"] = False
            if self._json_response.isChecked():
                result["json_response"] = True
            if self._context_turns.value() > 0:
                result["context_turns"] = self._context_turns.value()
            if self._input_price.value() > 0:
                result["input_price"] = self._input_price.value()
            if self._output_price.value() > 0:
                result["output_price"] = self._output_price.value()

            overrides = {}
            for key, (cb, _row, widget) in self._adv_rows.items():
                if cb.isChecked():
                    val = widget.value()
                    if isinstance(widget, QDoubleSpinBox):
                        val = round(val, 2)
                    overrides[key] = val
            if overrides:
                result["overrides"] = overrides

            ok, data = self._parse_extra_body()
            if ok and data:
                result["extra_body"] = data

        elif tl_type == "baidu":
            result["app_id"] = self._baidu_app_id.text().strip()
            result["secret_key"] = self._baidu_secret_key.text().strip()

        elif tl_type == "tencent":
            result["secret_id"] = self._tencent_secret_id.text().strip()
            result["secret_key"] = self._tencent_secret_key.text().strip()
            result["region"] = self._tencent_region.currentData()

        elif tl_type == "youdao":
            result["app_key"] = self._youdao_app_key.text().strip()
            result["app_secret"] = self._youdao_app_secret.text().strip()

        elif tl_type == "deepl":
            result["api_key"] = self._deepl_api_key.text().strip()

        return result


_I18N_DIR = Path(__file__).parent / "i18n"


def _changelog_to_html(text: str) -> str:
    """Convert CHANGELOG.md subset to HTML (headings, bold, lists)."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            continue  # skip file title
        elif stripped.startswith("- "):
            item = stripped[2:]
            item = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", item)
            item = re.sub(r"`(.+?)`", r"<code>\1</code>", item)
            lines.append(f"<li>{item}</li>")
        elif stripped:
            lines.append(f"<p>{stripped}</p>")
    return "\n".join(lines)


def _load_latest_changelog() -> tuple[str, str]:
    """Return (first_h2_title, html) for the latest changelog. Uses i18n lang."""
    from i18n import get_lang
    lang = get_lang()
    path = _I18N_DIR / f"CHANGELOG_{lang}.md"
    if not path.exists():
        path = _I18N_DIR / "CHANGELOG_en.md"
    if not path.exists():
        return "", ""
    text = path.read_text("utf-8")
    # First H2 (## date) is the latest entry and serves as the tracking key
    m = re.search(r"^## (.+)$", text, re.MULTILINE)
    if not m:
        return "", ""
    title = m.group(1).strip()
    # Drop the top-level file heading (# Title) — keep everything from first H2 onwards
    body = text[m.start():]
    return title, _changelog_to_html(body)


